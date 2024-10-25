import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path

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
        transform=None,
        subset: Optional[List[str]] = None
    ):
        """
        Args:
            root_dir: 数据根目录
            labels_file: 标签文件路径 (labels.json)
            norm_logs_file: 规范化日志文件路径 (norm_logs.json)
            transform: 可选的图像变换
            subset: 可选的主体ID列表，用于划分训练/验证集
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # 加载标签和日志
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
            
        with open(norm_logs_file, 'r') as f:
            self.norm_logs = json.load(f)
            
        # 构建数据索引
        self.samples = []
        for subject_id, slices in self.labels.items():
            if subset and subject_id not in subset:
                continue
                
            for slice_id, label in slices.items():
                self.samples.append({
                    'subject_id': subject_id,
                    'slice_id': slice_id,
                    'label': label
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sample = self.samples[idx]
        subject_id = sample['subject_id']
        slice_id = sample['slice_id']
        slice_num = int(slice_id.split('_')[1])  # 从'slice_X'提取数字
        
        # 加载光流图像序列
        frames = []
        for frame_idx in range(30):  # 每秒30帧
            img_path = self.root_dir / 'OpticalFlows' / f'sub_{subject_id}_sclice_{slice_num}_frame_{frame_idx}_optical.png'
            frame = cv2.imread(str(img_path))
            if frame is None:
                raise ValueError(f"无法加载图像: {img_path}")
            
            # 转换为RGB并归一化
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0
            
            if self.transform:
                frame = self.transform(frame)
            
            frames.append(frame)
        
        # 将帧堆叠为一个张量 [30, H, W, C]
        frames_tensor = torch.FloatTensor(np.stack(frames))
        
        # 获取标签
        label_tensor = torch.tensor(sample['label'], dtype=torch.float32)
        
        # 可选：从norm_logs中获取额外特征
        extra_features = self.norm_logs[subject_id][slice_id]
        
        metadata = {
            'subject_id': subject_id,
            'slice_id': slice_id,
            'extra_features': extra_features
        }
        
        return frames_tensor, label_tensor, metadata

def create_data_loaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_subjects: Optional[List[str]] = None,
    val_subjects: Optional[List[str]] = None,
    transform=None
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器

    Args:
        root_dir: 数据根目录
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        train_subjects: 训练集的主体ID列表
        val_subjects: 验证集的主体ID列表
        transform: 可选的图像变换

    Returns:
        训练数据加载器和验证数据加载器的元组
    """
    labels_file = os.path.join(root_dir, 'labels.json')
    norm_logs_file = os.path.join(root_dir, 'norm_logs.json')
    
    # 创建训练集
    train_dataset = VRMotionSicknessDataset(
        root_dir=root_dir,
        labels_file=labels_file,
        norm_logs_file=norm_logs_file,
        transform=transform,
        subset=train_subjects
    )
    
    # 创建验证集
    val_dataset = VRMotionSicknessDataset(
        root_dir=root_dir,
        labels_file=labels_file,
        norm_logs_file=norm_logs_file,
        transform=transform,
        subset=val_subjects
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# 使用示例
if __name__ == '__main__':
    # 定义一些图像变换
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])
    
    # 划分训练集和验证集的主体
    all_subjects = ['TYR', 'XSJ', 'CM', 'TX', 'HZ', 'CYL', 'GKW', 'LMH', 'WJX', 
                'CWG', 'SHQ', 'YHY', 'YCR']
    train_subjects = all_subjects[:-3]  # 最后3个主体用于验证
    val_subjects = all_subjects[-3:]
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        root_dir='.',
        batch_size=16,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        transform=transform
    )
    
    # 测试数据加载
    for frames, labels, metadata in train_loader:
        print(f"Frames batch shape: {frames.shape}")  # [B, 30, H, W, C]
        print(f"Labels batch shape: {labels.shape}")  # [B]
        print(f"Sample metadata: {metadata}")
        break