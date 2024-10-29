import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from prefetch_generator import BackgroundGenerator
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data._utils.collate import default_collate



class DataLoaderX(DataLoader):
    """优化的DataLoader
    """
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class VRMotionSicknessDataset(Dataset):
    """
    每个样本包含:
    - 30帧的原始图像序列 (1秒的数据)
    - 30帧光流图像序列 (1秒的数据)
    - (*, 30)帧的运动数据（日志） 不完全到30帧 因为unity的log有时间差
    - 对应的标签 (0或1表示是否有眩晕症状)
    """
    def __init__(
        self,
        root_dir: str,
        labels_file: str,
        norm_logs_file: str,
        subset: Optional[List[str]] = None,
        cache_size: int = 100,
        num_workers: int = 4,
        prefetch: bool = True
    ):
        """
        Args:
            root_dir: 数据根目录
            labels_file: 标签文件路径 (labels.json)
            norm_logs_file: 规范化日志文件路径 (norm_logs.json)
            subset: 可选的主体ID列表，用于划分训练/验证集
        """
        self.root_dir = Path(root_dir)
        self.cache_size = cache_size
        self.num_workers = num_workers
        self.prefetch = prefetch

        self.cache = {}
        self.cache_queue = []

        self.labels = self._load_json(labels_file)
        self.norm_logs = self._load_json(norm_logs_file)
        self.samples = self._build_samples(subset)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        if self.prefetch:
            self._prefetch_data()
        
    
    def _load_json(self, file_path: str) -> Dict:
        """使用内存映射加载JSON"""
        full_path = os.path.join(self.root_dir, file_path)
        try:
            with open(full_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"无法加载JSON文件: {full_path}")
    
    def _build_samples(self, subset: Optional[List[str]]) -> List[Dict]:
        """构建数据索引"""
        samples = []
        for subject_id, slices in self.labels.items():
            if subset and subject_id not in subset:
                continue
            
            for slice_id, label in slices.items():
                samples.append({
                    'subject_id': subject_id,
                    'slice_id': slice_id,
                    'label': label
                })
        return samples
    
    def _load_image(self, img_path: Path) -> np.ndarray:
        """加载和预处理图像 （ 单个）"""
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise ValueError(f"无法加载图像: {img_path}")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def _load_frame_sequence(self, subject_id: str, slice_id: str) -> np.ndarray:
        """加载图像序列"""
        futures_optical = []
        futures_original = []
        # 本来应该是 ... _{slice_id}_frame_ ... 
        # 数据集文件保存的时候出现了一个typo: slice -> sclice
        # 这里需要重新分割字符串
        slice_id = slice_id[-1]
        # todo 这里只有0-28帧 可能是数据集生成的时候的bug？
        for frame_idx in range(29):
            img_path_optical = self.root_dir / f'sub_{subject_id}_sclice_{slice_id}_frame_{frame_idx}_optical.png'
            img_path_original = self.root_dir / f'sub_{subject_id}_sclice_{slice_id}_frame_{frame_idx}_original.png'

            # submit方法提交load image任务给线程池
            futures_optical.append(self.executor.submit(self._load_image, img_path_optical))
            futures_original.append(self.executor.submit(self._load_image, img_path_original))

        # 获取加载结果
        frames_optical = [fut.result() for fut in futures_optical]
        frames_original = [fut.result() for fut in futures_original]

        return np.stack(frames_optical), np.stack(frames_original)

    def _manage_cache(self, idx: int, data: tuple):
        """缓存管理"""
        if len(self.cache) >= self.cache_size:
            # 删除最旧的缓存
            self.cache.pop(self.cache_queue.pop(0))
        self.cache[idx] = data
        self.cache_queue.append(idx)
    
    def _prefetch_data(self):
        """预加载数据"""
        prefetch_size = min(self.cache_size, len(self.samples))
        for idx in range(prefetch_size):
            self._load_sample(idx)
    
    def _load_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
            顺序：光流图像序列，原始图像序列，运动特征，标签
        """
        sample = self.samples[idx]
        subject_id = sample['subject_id']
        slice_id = sample['slice_id']

        # 并行加载图像
        frames_optical, frames_original = self._load_frame_sequence(subject_id, slice_id)
        frames_optical_tensor = torch.FloatTensor(frames_optical)
        frames_original_tensor = torch.FloatTensor(frames_original)

        motion_features = self.norm_logs[subject_id][slice_id]

        motion_data = {
            'subject_id': subject_id,
            'slice_id': slice_id,
            'motion_features': motion_features
        }

        label_tensor = torch.tensor(sample['label'], dtype=torch.float32)

        return frames_optical_tensor, frames_original_tensor, motion_data, label_tensor 



    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if idx in self.cache:
            return self.cache[idx]
        
        data = self._load_sample(idx)
        self._manage_cache(idx, data)
        return data
    def clear_cache(self):
        self.cache.clear()
        self.cache_queue.clear()
    

def get_vr_dataloader(
    dataset: VRMotionSicknessDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoaderX:
    """创建优化的DataLoader"""
    return DataLoaderX(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=default_collate
    )

# 使用示例
if __name__ == '__main__':
    # 划分训练集和验证集的主体
    all_subjects = ['TYR', 'XSJ', 'CM', 'TX', 'HZ', 'CYL', 'GKW', 'LMH', 'WJX', 
                'CWG', 'SHQ', 'YHY', 'YCR']
    train_subjects = all_subjects[:-3]  # 最后3个主体用于验证
    val_subjects = all_subjects[-3:]

    # todo use os.path.join
    dataset = VRMotionSicknessDataset(
        root_dir="./data",
        labels_file="labels.json",
        norm_logs_file="norm_logs.json",
        subset=train_subjects,
        cache_size=100,
        num_workers=4,
        prefetch=True
    )
    dataloader = get_vr_dataloader(
        dataset=dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )
    
    # 测试数据加载
    for frames_optical_tensor, frames_original_tensor, motion_data, label_tensor  in dataloader:
        print(f"Frames batch shape: {frames_optical_tensor.shape}")  # [B, 30, H, W, C]
        print(f"Frames(R) batch shape: {frames_original_tensor.shape}")  # [B, 30, H, W, C]
        print(f"Sample metadata: {motion_data}") # [B , Dict ?]
        print(f"Labels batch shape: {label_tensor.shape}")  # [B]
        break