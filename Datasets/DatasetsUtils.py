import torch
from typing import Optional, Dict
from torch import nn
from typing import Union
from DataClass import BatchData, MotionFeatureConfig
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

def set_device(tensor_or_module: Union[torch.Tensor, nn.Module], device: str) -> Union[torch.Tensor, nn.Module]:
    """将tensor或module移动到指定设备"""
    if isinstance(tensor_or_module, (torch.Tensor, nn.Module)):
        return tensor_or_module.to(device)
    return tensor_or_module

class EnsureThreeChannels(nn.Module):
    """确保图像是三通道的自定义转换类"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
    
def move_to_device(batch: BatchData, device: str) -> BatchData:
    """将 batch 数据移动到指定设备"""
    return BatchData(
        frames_optical=batch.frames_optical.to(device),
        frames_original=batch.frames_original.to(device),
        motion_features=batch.motion_features.to(device),
        motion_metadata=batch.motion_metadata,
        labels=batch.labels.to(device),
        mask=batch.mask.to(device)
    )

class CollateProcessor:
    def __init__(self, feature_config: Optional[MotionFeatureConfig] = None):
        self.feature_config = feature_config or MotionFeatureConfig()
        self.feature_stats = {}
        
    def _extract_motion_features(self, motion_data: Dict) -> torch.Tensor:
        """从运动数据中提取特征 - 优化版本"""
        features = torch.zeros((30, len(self.feature_config.feature_names)), dtype=torch.float32)
        motion_features = motion_data['motion_features']
        
        for frame_idx in range(30):
            frame_key = f'frame_{frame_idx}'
            if frame_key not in motion_features:
                continue
            frame_data = motion_features[frame_key]
            for feat_idx, feature_name in enumerate(self.feature_config.feature_names):
                value = frame_data.get(feature_name)
                if isinstance(value, torch.Tensor):
                    features[frame_idx, feat_idx] = value.item()
        return features

    
    def _normalize_features(self, features: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """标准化特征"""
        if not self.feature_config.normalize:
            return features
            
        B, T, F = features.shape
        features_flat = features.reshape(-1, F)
        
        if update_stats:
            mean = features_flat.mean(dim=0)
            std = features_flat.std(dim=0)
            
            if not self.feature_stats:
                self.feature_stats['mean'] = mean
                self.feature_stats['std'] = std
            else:
                momentum = 0.1
                self.feature_stats['mean'] = (1 - momentum) * self.feature_stats['mean'] + momentum * mean
                self.feature_stats['std'] = (1 - momentum) * self.feature_stats['std'] + momentum * std
        
        mean = self.feature_stats.get('mean', mean)
        std = self.feature_stats.get('std', std)
        std = torch.clamp(std, min=1e-6)
        
        normalized = (features_flat - mean) / std
        return normalized.reshape(B, T, F)
    
    def __call__(self, batch: List[Tuple]) -> BatchData:
        """处理批量数据，直接处理元组形式的数据"""
        # 解包batch中的元组数据
        batch_unpacked = [
            (
                frames_opt,  # frames_optical
                frames_orig, # frames_original
                motion_data, # motion_data
                label       # label
            ) for frames_opt, frames_orig, motion_data, label in batch
        ]
        
        # 分别提取每种数据
        frames_optical = [item[0] for item in batch_unpacked]
        frames_original = [item[1] for item in batch_unpacked]
        motion_data = [item[2] for item in batch_unpacked]
        labels = [item[3] for item in batch_unpacked]
        
        # 获取序列长度并创建mask
        lengths = [f.size(0) for f in frames_optical]
        max_len = max(lengths)
        batch_size = len(batch)
        
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        
        frames_optical_padded = pad_sequence(frames_optical, batch_first=True)
        frames_original_padded = pad_sequence(frames_original, batch_first=True)
        
        motion_features = []
        motion_metadata = {
            'subject_ids': [],
            'slice_ids': []
        }
        
        for data in motion_data:
            motion_metadata['subject_ids'].append(data['subject_id'])
            motion_metadata['slice_ids'].append(data['slice_id'])
            features = self._extract_motion_features(data)
            motion_features.append(features)
        
        motion_features_padded = pad_sequence(motion_features, batch_first=True)
        motion_features_normalized = self._normalize_features(motion_features_padded)
        
        device = batch[0][0].device

        return BatchData(
            frames_optical=frames_optical_padded.to(device),
            frames_original=frames_original_padded.to(device),
            motion_features=motion_features_normalized.to(device),
            motion_metadata=motion_metadata,
            labels=torch.stack(labels).to(device),
            mask=mask.to(device)
        )

class DeviceCollator:
    def __init__(self, processor: CollateProcessor, device: str):
        self.processor = processor
        self.device = device
        
    def __call__(self, batch):
        processed = self.processor(batch)
        return move_to_device(processed, self.device)
    

class DataPrefetcher:
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader  # 保留loader引用，确保不会在__init__中将其转换为迭代器
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self._loader_iter = iter(self.loader)  # 定义单独的迭代器
        self._preload()

        self.collate_fn = loader.collate_fn
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory
        self.drop_last = loader.drop_last

    def _preload(self):
        try:
            self.next_batch = next(self._loader_iter)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            # 将批次数据移动到GPU并使用异步加载
            if isinstance(self.next_batch, BatchData):
                self.next_batch = move_to_device(self.next_batch, self.device)
            else:
                for k, v in self.next_batch.items():
                    if torch.is_tensor(v):
                        self.next_batch[k] = v.to(self.device, non_blocking=True)

    def __iter__(self):
        return self
    
    def __next__(self):
        # 等待当前CUDA流同步
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        self._preload()  # 预加载下一个批次
        return batch

