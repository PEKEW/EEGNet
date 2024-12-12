import torchvision.transforms as transforms
import re
from pathlib import Path
import mne
import tarfile
import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import warnings
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


class VRSicknessDataset(Dataset):

    SUB_ID_PATTERN = re.compile(r'(\w+)_combined')
    SLICE_ID_PATTERN_TYPO = re.compile(r'sclice_(\d+)')
    SLICE_ID_PATTERN = re.compile(r'slice_(\d+)')
    FRAME_ID_PATTERN = re.compile(r'frame_(\d+)')
    MOTION_LOG_PATH_STR = 'norm_logs.json'
    FRAME_DIR_STR = 'frame_archives'
    EEG_DIR_STR = 'EEGData'
    LABEL_DIR_STR = 'labels.json'

    def __init__(self, root_dir, transform=None, mod: list = ['eeg', 'video', 'log']):
        self.root_dir = Path(root_dir)
        self.frame_dir = self.root_dir / self.FRAME_DIR_STR
        self.eeg_dir = self.root_dir / self.EEG_DIR_STR
        self.transform = transform
        self.mod = mod
        with open(self.root_dir / self.MOTION_LOG_PATH_STR, 'r') as f:
            self.motion_data = json.load(f)
        with open(self.root_dir / self.LABEL_DIR_STR, 'r') as f:
            self.labels = json.load(f)
        self.samples = self._get_valid_samples()

        if 'eeg' in self.mod:
            self.eeg_data = {sub_id: {} for sub_id, _ in self.samples}
            self._load_all_eeg()

    def _get_valid_samples(self):
        def get_slice_id(text, pattern):
            match = pattern.search(text)
            return match.group(1) if match else None
        valid_samples = []
        for frame_tar in self.frame_dir.glob('*_combined.tar'):
            sub_id_match = get_slice_id(frame_tar.name, self.SUB_ID_PATTERN)
            if not sub_id_match:
                continue
            sub_id = sub_id_match
            eeg_tar = self.eeg_dir / f'{sub_id}.tar'
            if not eeg_tar.exists() or sub_id not in self.motion_data:
                continue
            with tarfile.open(frame_tar, 'r') as tar:
                frame_files = [f for f in tar.getnames() if 'optical.png' in f]
            with tarfile.open(eeg_tar, 'r') as tar:
                eeg_files = [f for f in tar.getnames() if f.endswith('.set')]
            optical_frame_slices = {
                slice_id
                for f in frame_files
                if (slice_id := get_slice_id(f, self.SLICE_ID_PATTERN_TYPO))
            }
            eeg_slices = {
                slice_id
                for f in eeg_files
                if (slice_id := get_slice_id(f, self.SLICE_ID_PATTERN))
            }
            motion_slices = {
                slice_id
                for k in self.motion_data[sub_id].keys()
                if (slice_id := get_slice_id(k, self.SLICE_ID_PATTERN))
            }
            valid_slices = optical_frame_slices & eeg_slices & motion_slices
            valid_samples.extend((sub_id, slice_id) for slice_id in valid_slices)
                
        # TODO: important cut samples for test
        # return valid_samples
        return valid_samples[0:500]

    def _load_frames(self, sub_id, slice_id):
        def get_frame_id(name: str) -> int:
            match = self.FRAME_ID_PATTERN.search(name)
            if match is None:
                raise ValueError(f"Invalid frame name format: {name}")
            numeric_id = match.group(1)
            if numeric_id is None:
                raise ValueError(f"Could not extract frame ID from: {name}")
            return int(numeric_id)
        
        tar_path = self.frame_dir / f'{sub_id}_combined.tar'
        frames = {'optical': [], 'original': []}
        pattern = f'sub_{sub_id}_sclice_{slice_id}_frame_'
        
        with tarfile.open(tar_path, 'r') as tar:
            frame_members = []
            for m in tar.getmembers():
                if pattern not in m.name:
                    continue
                try:
                    frame_id = get_frame_id(m.name)
                    frame_members.append(m)
                except ValueError:
                    continue
            if not frame_members:
                raise ValueError(f"No frames found for sub_{sub_id}, slice_{slice_id}")
            frame_members.sort(key=lambda x: get_frame_id(x.name))
            for member in frame_members:
                frame_file = tar.extractfile(member)
                if frame_file is None:
                    continue
                try:
                    image = Image.open(frame_file).convert('RGB')
                    transform = self.transform or transforms.Compose([
                        transforms.ToTensor()
                    ])
                    image_tensor = transform(image)
                    if 'optical.png' in member.name:
                        frames['optical'].append(image_tensor)
                    elif 'original.png' in member.name:
                        frames['original'].append(image_tensor)
                finally:
                    frame_file.close()
        # if not frames['optical'] or not frames['original']:
        #     raise ValueError(f"Missing frames for sub_{sub_id}, slice_{slice_id}")
        optical_stack = torch.stack(frames['optical'])
        original_stack = torch.stack(frames['original'])
        if len(optical_stack) != len(original_stack):
            raise ValueError(
                f"Mismatched frame counts for sub_{sub_id}, slice_{slice_id}: "
                f"optical={len(optical_stack)}, original={len(original_stack)}"
            )
        
        return optical_stack, original_stack

    def _load_all_eeg(self):
        sub_id_set = set(sub_id for sub_id, _ in self.samples)
        for sub_id in sub_id_set:
            tar_path = self.eeg_dir / f'{sub_id}.tar'
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path='/tmp')
            slice_id_set = set(slice_id for s_id,
                            slice_id in self.samples if s_id == sub_id)
            for slice_id in slice_id_set:
                set_name = f'{sub_id}_slice_{slice_id}.set'
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        raw = mne.io.read_raw_eeglab(
                            f'/tmp/{set_name}', preload=True)
                    data = raw.get_data()
                    data_tensor = torch.FloatTensor(data)
                    _mean = data_tensor.mean(dim=1, keepdim=True)
                    _std = data_tensor.std(dim=1, keepdim=True)
                    normalized_data = (data_tensor - _mean) / (_std + 1e-10)
                    self.eeg_data[sub_id][slice_id] = normalized_data
                finally:
                    os.remove(f'/tmp/{set_name}')

    def _load_motion(self, sub_id, slice_id):
        def sort_frame_ids(x):
            match = self.FRAME_ID_PATTERN.search(x)
            if match is None:
                raise ValueError(f"Invalid frame name format: {x}")
            return int(match.group(1))
        padding_mod = 'last'  # last | first
        slice_data = self.motion_data[sub_id][f'slice_{slice_id}']
        frame_ids = sorted(slice_data.keys(), key=sort_frame_ids)

        motion_features = []
        for frame_id in frame_ids:
            frame_data = slice_data[frame_id]
            pos = eval(frame_data['pos'])
            features = [
                float(frame_data['time_']),
                float(frame_data['speed']),
                float(frame_data['acceleration']),
                float(frame_data['rotation_speed']),
                int(frame_data['is_sickness']),
                int(frame_data['complete_sickness']),
                pos[0], pos[1], pos[2]
            ]
            motion_features.append(features)
        
        num_frames = len(motion_features)
        if num_frames < 30:
            padding_features = (
                motion_features[-1] if padding_mod == 'last'
                else motion_features[0] if padding_mod == 'first'
                else None
            )
            
            if padding_features is None:
                raise NotImplementedError(f"Unsupported padding mode: {padding_mod}")
            motion_features.extend([padding_features] * (30 - num_frames))
        return torch.FloatTensor(motion_features)

    def _load_label(self, sub_id, slice_id):
        return torch.FloatTensor([self.labels[sub_id][f'slice_{slice_id}']])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sub_id, slice_id = self.samples[idx]

        try:
            if 'original' in self.mod:
                optical_frames, original_frames = self._load_frames(
                    sub_id, slice_id)
            elif 'optical' in self.mod:
                optical_frames, original_frames = self._load_frames(
                    sub_id, slice_id)
            else:
                optical_frames, original_frames = None, None
            if 'motion' in self.mod:
                motion_data = self._load_motion(sub_id, slice_id)
            else:
                motion_data = None

            labels = self._load_label(sub_id, slice_id)

            return {
                'sub_id': sub_id,
                'slice_id': slice_id,
                'optical': optical_frames,
                'original': original_frames,
                'eeg': self.eeg_data[sub_id][slice_id] if 'eeg' in self.mod else None,
                'motion': motion_data,
                'label': labels
            }

        except Exception as e:
            print(f"Error loading sample (sub_{sub_id}, slice_{slice_id}): {str(e)}")
            return None
