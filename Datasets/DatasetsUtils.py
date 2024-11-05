import torch # tag
from typing import Optional, Dict
from torch import nn
from typing import Union
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List

FPS = 30-1

class SequenceCollator:
    def __init__(self, sequence_length=None, padding_mode='zero'):
        self.sequence_length = sequence_length
        self.padding_mode = padding_mode
        
    def __call__(self, batch):
        """
        可序列化的 collate 函数
        """
        # 过滤掉 None 值
        batch = [b for b in batch if b is not None]
        if not batch:
            return []
        
        # 获取当前批次中最长的序列长度
        max_motion_length = max(s['motion'].size(0) for s in batch)
        max_frame_length = max(s['optical_frames'].size(0) for s in batch)
        
        # 如果指定了序列长度，使用指定长度
        if self.sequence_length is not None:
            max_motion_length = self.sequence_length
            max_frame_length = self.sequence_length
        
        processed_batch = {
            'sub_id': [],
            'slice_id': [],
            'optical_frames': [],
            'original_frames': [],
            'eeg': [],
            'motion': [],
            'label': [],
            'lengths': []
        }
        
        for sample in batch:
            processed_batch['sub_id'].append(sample['sub_id'])
            processed_batch['slice_id'].append(sample['slice_id'])
            
            # 处理 motion 数据
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

            
            # 处理帧数据
            optical_frames = sample['optical_frames']
            original_frames = sample['original_frames']
            
            if optical_frames.size(0) > max_frame_length:
                optical_frames = optical_frames[:max_frame_length]
                original_frames = original_frames[:max_frame_length]
            elif optical_frames.size(0) < max_frame_length:
                pad_length = max_frame_length - optical_frames.size(0)
                if self.padding_mode == 'zero':
                    pad_optical = torch.zeros((pad_length, *optical_frames.size()[1:]), dtype=optical_frames.dtype)
                    pad_original = torch.zeros((pad_length, *original_frames.size()[1:]), dtype=original_frames.dtype)
                else:  # repeat mode
                    pad_optical = optical_frames[-1].unsqueeze(0).repeat(pad_length, 1, 1, 1)
                    pad_original = original_frames[-1].unsqueeze(0).repeat(pad_length, 1, 1, 1)
                
                optical_frames = torch.cat([optical_frames, pad_optical], dim=0)
                original_frames = torch.cat([original_frames, pad_original], dim=0)
            
            processed_batch['optical_frames'].append(optical_frames)
            processed_batch['original_frames'].append(original_frames)
            processed_batch['eeg'].append(sample['eeg'])
        
        # 堆叠处理后的张量
        return {
            'sub_id': processed_batch['sub_id'],
            'slice_id': processed_batch['slice_id'],
            'optical_frames': torch.stack(processed_batch['optical_frames']),
            'original_frames': torch.stack(processed_batch['original_frames']),
            'eeg': torch.stack(processed_batch['eeg']),
            'motion': torch.stack(processed_batch['motion']),
            'lengths': torch.tensor(processed_batch['lengths']),
            'label': torch.stack(processed_batch['label'])
        }

class EnsureThreeChannels(nn.Module):
    """确保图像是三通道的自定义转换类"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'