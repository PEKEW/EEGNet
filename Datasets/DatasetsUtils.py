import torch # tag
from typing import Optional, Dict
from torch import nn
from typing import Union
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List

FPS = 30-1

class SequenceCollator:
    def __init__(self, sequence_length=None, padding_mode='zero', include=['eeg','label','optical', 'original']):
        self.sequence_length = sequence_length
        self.padding_mode = padding_mode
        self.include = include
        
    def __call__(self, batch):
        """
        可序列化的 collate 函数
        """
        # 过滤掉 None 值
        batch = [b for b in batch if b is not None]
        if not batch:
            return []
        
        max_motion_length = 0
        max_optical_frame_length = 0
        max_original_frame_length = 0
        if 'motion' in self.include:
            max_motion_length = max(s['motion'].size(0) for s in batch)
        if 'optical' in self.include:
            max_optical_frame_length = max(s['optical_frames'].size(0) for s in batch)
        if 'original' in self.include:
            max_original_frame_length = max(s['original_frames'].size(0) for s in batch)
        
        # 如果指定了序列长度，使用指定长度
        if self.sequence_length is not None:
            max_motion_length = self.sequence_length
            max_optical_frame_length = self.sequence_length
            max_original_frame_length = self.sequence_length
        
        processed_batch = {
            'sub_id': [],
            'slice_id': [],
            'label': [],
            'lengths': []
        }

        if 'eeg' in self.include:
            processed_batch['eeg'] = []
        if 'optical' in self.include:
            processed_batch['optical_frames'] = []
        if 'original' in self.include:
            processed_batch['original_frames'] = []
        if 'motion' in self.include:
            processed_batch['motion'] = []

        for sample in batch:
            processed_batch['sub_id'].append(sample['sub_id'])
            processed_batch['slice_id'].append(sample['slice_id'])
            
            # 处理label
            labels = sample['label']
            curr_length = labels.size(0)
            if curr_length > max_motion_length:
                labels = labels[:max_motion_length]
            elif curr_length < max_motion_length:
                pad_length = max_motion_length - curr_length
                pad = labels[-1].repeat(pad_length)
                labels = torch.cat([labels, pad])
            processed_batch['label'].append(labels)

            if 'motion' in self.include:
                motion_data = sample['motion']
                curr_length = motion_data.size(0)
                processed_batch['lengths'].append(curr_length)
                
                if curr_length > max_motion_length:
                    motion_data = motion_data[:max_motion_length]
                elif curr_length < max_motion_length:
                    pad_length = max_motion_length - curr_length
                    if self.padding_mode == 'zero':
                        pad = torch.zeros((pad_length, motion_data.size(1)), dtype=motion_data.dtype)
                    else:  # repeat mode
                        pad = motion_data[-1].unsqueeze(0).repeat(pad_length, 1)
                    motion_data = torch.cat([motion_data, pad], dim=0)
                processed_batch['motion'].append(motion_data)


            if 'original' in self.include:
                original_frames = sample['original_frames']
                if original_frames.size(0) > max_original_frame_length:
                    original_frames = original_frames[:max_optical_frame_length]
                elif original_frames.size(0) < max_original_frame_length:
                    pad_length = max_original_frame_length - original_frames.size(0)
                    if self.padding_mode == 'zero':
                        pad_original = torch.zeros((pad_length, *original_frames.size()[1:]), dtype=original_frames.dtype)
                    else:
                        pad_original = original_frames[-1].unsqueeze(0).repeat(pad_length, 1, 1, 1)
                    original_frames = torch.cat([original_frames, pad_original], dim=0)
                processed_batch['original_frames'].append(original_frames)

            if 'optical' in self.include:
                optical_frames = sample['optical_frames']
                if optical_frames.size(0) > max_optical_frame_length:
                    optical_frames = optical_frames[:max_optical_frame_length]
                elif optical_frames.size(0) < max_optical_frame_length:
                    pad_length = max_optical_frame_length - optical_frames.size(0)
                    if self.padding_mode == 'zero':
                        pad_optical = torch.zeros((pad_length, *optical_frames.size()[1:]), dtype=optical_frames.dtype)
                    else:  # repeat mode
                        pad_optical = optical_frames[-1].unsqueeze(0).repeat(pad_length, 1, 1, 1)
                    optical_frames = torch.cat([optical_frames, pad_optical], dim=0)
                processed_batch['optical_frames'].append(optical_frames)

            if 'eeg' in self.include:
                processed_batch['eeg'].append(sample['eeg'])
        
        res =  {
            'sub_id': processed_batch['sub_id'],
            'slice_id': processed_batch['slice_id'],
            'lengths': torch.tensor(processed_batch['lengths']),
            'label': torch.stack(processed_batch['label'])
        }
        if 'eeg' in self.include:
            res['eeg'] = torch.stack(processed_batch['eeg'])
        if 'optical' in self.include:
            res['optical_frames'] = torch.stack(processed_batch['optical_frames'])
        if 'original' in self.include:
            res['original_frames'] = torch.stack(processed_batch['original_frames'])
        if 'motion' in self.include:
            res['motion'] = torch.stack(processed_batch['motion'])
        return res

class EnsureThreeChannels(nn.Module):
    """确保图像是三通道的自定义转换类"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'