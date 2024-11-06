import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import warnings
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import os
import json
import tarfile
import mne
from pathlib import Path
import re
from pathlib import Path
import tarfile
import mne
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.multiprocessing as tmp
from Datasets.DatasetsUtils import SequenceCollator

class GenderSubjectSampler(Sampler):
    male_sub_list = ['TYR', 'XSJ', 'CM', 'SHQ', 'LMH', 'LZX']
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = []
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


class GenderSubjectSamplerMale(GenderSubjectSampler):
    def __init__(self, dataset):
        super().__init__(dataset)

        self.indices = [
            i for i, (sub_id, _) in enumerate(dataset.samples)
            if (sub_id in self.male_sub_list)
        ]
class GenderSubjectSamplerFemale(GenderSubjectSampler):
    def __init__(self, dataset):
        super().__init__(dataset)

        self.indices = [
            i for i, (sub_id, _) in enumerate(dataset.samples)
            if (sub_id not in self.male_sub_list)
        ]

class InterSubjectSampler(Sampler):
    """跨被试采样器"""
    def __init__(self, dataset, fold, sub_list, n_per, is_train=True):
        self.dataset = dataset
        self.fold = fold
        self.sub_list = sub_list
        self.n_subs = len(sub_list)
        self.n_per = n_per
        self.is_train = is_train
        
        val_start = n_per * fold
        val_end = min(n_per * (fold + 1), self.n_subs)
        val_subs = set(self.sub_list[val_start:val_end])
        
        self.indices = [
            i for i, (sub_id, _) in enumerate(dataset.samples)
            if (sub_id in val_subs) != is_train
        ]
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


class ExtraSubjectSampler(Sampler):
    def __init__(self, dataset, fold, sub_list, n_per, is_train=True):
        pass
    def __iter__(self):
        raise NotImplementedError("ExtraSubjectSampler is not implemented yet.")


class VRSicknessDataset(Dataset):

    SUB_ID_PATTERN = re.compile(r'(\w+)_combined')
    SLICE_ID_PATTERN_TYPO = re.compile(r'sclice_(\d+)')
    SLICE_ID_PATTERN = re.compile(r'slice_(\d+)')
    FRAME_ID_PATTERN = re.compile(r'frame_(\d+)')
    MOTION_LOG_PATH_STR = 'norm_logs.json'
    FRAME_DIR_STR = 'frame_archives'
    EEG_DIR_STR = 'EEGData'
    LABEL_DIR_STR = 'labels.json'

    def __init__(self, root_dir, transform=None, mod:list=['eeg', 'video', 'log']):
        self.root_dir = Path(root_dir)
        self.frame_dir = self.root_dir / self.FRAME_DIR_STR
        self.eeg_dir = self.root_dir / self.EEG_DIR_STR
        self.transform = transform
        self.mod = mod
        
        # 加载运动数据
        with open(self.root_dir / self.MOTION_LOG_PATH_STR, 'r') as f:
            self.motion_data = json.load(f)

        # 加载标签
        with open(self.root_dir / self.LABEL_DIR_STR, 'r') as f:
            self.labels = json.load(f)
            
        # 获取所有有效的样本
        self.samples = self._get_valid_samples()

        if 'eeg' in self.mod:
            self.eeg_data = {sub_id: {} for sub_id, _ in self.samples}
            self._load_all_eeg()
        
    def _get_valid_samples(self):
        """获取所有有效的样本对（sub_id, slice_id）"""
        valid_samples = []
        for frame_tar in self.frame_dir.glob('*_combined.tar'):
            sub_id = self.SUB_ID_PATTERN.search(frame_tar.name).group(1)
            eeg_tar = self.eeg_dir / f'{sub_id}.tar'
            if not eeg_tar.exists():
                continue
            if sub_id not in self.motion_data:
                continue
            with tarfile.open(frame_tar, 'r') as tar:
                frame_files = tar.getnames()
            with tarfile.open(eeg_tar, 'r') as tar:
                eeg_files = tar.getnames()
            original_frame_slices = set(self.SLICE_ID_PATTERN_TYPO.search(f).group(1)
                                for f in frame_files if 'optical.png' in f)
            optical_frame_slices = set(self.SLICE_ID_PATTERN_TYPO.search(f).group(1)
                                 for f in frame_files if 'optical.png' in f)
            eeg_slices = set(self.SLICE_ID_PATTERN.search(f).group(1)
                             for f in eeg_files if f.endswith('.set'))
            motion_slices = set(self.SLICE_ID_PATTERN.search(k).group(1)
                                for k in self.motion_data[sub_id].keys())
            valid_slices = optical_frame_slices & original_frame_slices & eeg_slices & motion_slices
            for slice_id in valid_slices:
                valid_samples.append((sub_id, slice_id))
        return valid_samples
    
    def _load_frames(self, sub_id, slice_id):
        """加载图片数据"""
        tar_path = self.frame_dir / f'{sub_id}_combined.tar'
        frames = {'optical': [], 'original': []}
        
        with tarfile.open(tar_path, 'r') as tar:
            pattern = f'sub_{sub_id}_sclice_{slice_id}_frame_'
            frame_members = [m for m in tar.getmembers() 
                        if pattern in m.name]
            if not frame_members:
                raise ValueError(f"No frames found for sub_{sub_id}, slice_{slice_id}")
            frame_members.sort(key=lambda x: int(self.FRAME_ID_PATTERN.search(x.name).group(1)))
            
            for member in frame_members:
                frame_file = tar.extractfile(member)
                image = Image.open(frame_file).convert('RGB')
                if self.transform:
                    image_tensor = self.transform(image)
                else:
                    default_transform = transforms.Compose([
                        transforms.ToTensor()
                    ])
                    image_tensor = default_transform(image)
                
                if 'optical.png' in member.name:
                    frames['optical'].append(image_tensor)
                elif 'original.png' in member.name:
                    frames['original'].append(image_tensor)
        
        if not frames['optical'] or not frames['original']:
            raise ValueError(f"Missing frames for sub_{sub_id}, slice_{slice_id}")
        
        optical_stack = torch.stack(frames['optical'])
        original_stack = torch.stack(frames['original'])
        
        assert len(optical_stack) == len(original_stack), \
            f"Mismatched frame counts for sub_{sub_id}, slice_{slice_id}"
        
        return optical_stack, original_stack
    

    def _load_all_eeg(self):
        """每个sub_id的tar都至少需要打开一次
        因此每次打开一个sub_id时
        就把所有的数据都加载出来放到内存
        这种方法只有sub_id次相关I/O
        """
        sub_id_set = set(sub_id for sub_id, _ in self.samples)
        for sub_id in sub_id_set:
            tar_path = self.eeg_dir / f'{sub_id}.tar'
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path='/tmp')
            slice_id_set = set(slice_id for s_id, slice_id in self.samples if s_id == sub_id)
            for slice_id in slice_id_set:
                set_name = f'{sub_id}_slice_{slice_id}.set'
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        raw = mne.io.read_raw_eeglab(f'/tmp/{set_name}', preload=True)
                    data = raw.get_data()
                    self.eeg_data[sub_id][slice_id] = torch.FloatTensor(data)
                finally:
                    os.remove(f'/tmp/{set_name}')

    def _load_eeg(self, sub_id, slice_id):
        """加载EEG数据"""
        tar_path = self.eeg_dir / f'{sub_id}.tar'
        set_name = f'{sub_id}_slice_{slice_id}.set'
        
        with tarfile.open(tar_path, 'r') as tar:
            # 提取.set文件到临时目录
            tar.extract(set_name, path='/tmp')
            
        try:
            # 读取EEG数据
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_eeglab(f'/tmp/{set_name}', preload=True)
            data = raw.get_data()
            return torch.FloatTensor(data)
        finally:
            # 清理临时文件
            os.remove(f'/tmp/{set_name}')
    
    def _load_motion(self, sub_id, slice_id):
        """加载运动数据"""
        slice_data = self.motion_data[sub_id][f'slice_{slice_id}']
        
        # 按帧ID排序
        frame_ids = sorted(slice_data.keys(), key=lambda x: int(self.FRAME_ID_PATTERN.search(x).group(1)))
        
        motion_features = []
        for frame_id in frame_ids:
            frame_data = slice_data[frame_id]
            # 解析位置字符串
            pos = eval(frame_data['pos'])  # 将字符串"(x,y,z)"转换为元组
            
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
            
        return torch.FloatTensor(motion_features)
    
    def _load_label(self, sub_id, slice_id):
        """加载标签"""
        return torch.FloatTensor([self.labels[sub_id][f'slice_{slice_id}']])


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sub_id, slice_id = self.samples[idx]
        
        try:
            if 'video' in self.mod:
                optical_frames, original_frames = self._load_frames(sub_id, slice_id)
            else:
                optical_frames, original_frames = None, None
            if 'log' in self.mod:
                motion_data = self._load_motion(sub_id, slice_id)
            else:
                motion_data = None

            labels = self._load_label(sub_id, slice_id)
            
            return {
                'sub_id': sub_id,
                'slice_id': slice_id,
                'optical_frames': optical_frames,
                'original_frames': original_frames,
                'eeg': self.eeg_data[sub_id][slice_id],
                'motion': motion_data,
                'label': labels
            }
        
        except Exception as e:
            print(f"Error loading sample (sub_{sub_id}, slice_{slice_id}): {str(e)}")
            return None
        
    @staticmethod
    def _compute_statistics(original_frames, optical_frames, eeg, motion):
        """优化的统计计算"""
        with torch.no_grad():
            return {
                'original_frames_stats': {
                    'mean': original_frames.mean().item(),
                    'std': original_frames.std().item(),
                    'min': original_frames.min().item(),
                    'max': original_frames.max().item()
                },
                'optical_frames_stats': {
                    'mean': optical_frames.mean().item(),
                    'std': optical_frames.std().item(),
                    'min': optical_frames.min().item(),
                    'max': optical_frames.max().item()
                },
                'eeg_stats': {
                    'mean': eeg.mean().item(),
                    'std': eeg.std().item(),
                    'min': eeg.min().item(),
                    'max': eeg.max().item()
                },
                'motion_stats': {
                    'mean': motion.mean().item(),
                    'std': motion.std().item(),
                    'min': motion.min().item(),
                    'max': motion.max().item()
                }
            }


def get_dataloader(root_dir, batch_size=32, num_workers=4, sequence_length=None, padding_mode='zero'):
    """创建数据加载器
    
    Args:
        sequence_length: 目标序列长度，None表示使用批次中最长序列的长度
        padding_mode: 填充模式，'zero' 或 'repeat'
    """
    dataset = VRSicknessDataset(root_dir=root_dir)
    collator = SequenceCollator(sequence_length, padding_mode)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return dataloader

# 用法
def train_loop(dataloader, model, device):
    for batch in dataloader:
        # 批量转移到GPU，而不是单个样本转移
        optical_frames = batch['optical_frames'].to(device, non_blocking=True)
        original_frames = batch['original_frames'].to(device, non_blocking=True)
        eeg = batch['eeg'].to(device, non_blocking=True)
        motion = batch['motion'].to(device, non_blocking=True)
        label = batch['label'].to(device, non_blocking=True)




# if __name__ == "__main__":
#     # 设置多进程启动方法为 spawn
#     if tmp.get_start_method(allow_none=True) is None:
#         tmp.set_start_method('spawn')

#     image_dir = "data/images"
#     eeg_dir = "data/eeg"
#     motion_dir = "data/motion"
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     dataloader = get_dataloader(
#         root_dir='/home/pekew/code/EEGNet/data',
#         batch_size=32,
#         sequence_length=None,  # 使用批次中最长序列长度
#         padding_mode='repeat')

#     train_loop(dataloader, None, device)