# VR动作晕眩数据集加载器文档

## 概述
`VRMotionSicknessDataset` 和相关的数据加载器提供了一种高效的方式来加载和处理VR动作晕眩数据，包括视频帧、脑电信号(EEG)和运动特征。本文档详细说明如何使用数据加载器及其配置选项。

## 快速开始
```python
from vr_dataset import VRMotionSicknessDataset, get_vr_dataloader, DatasetConfig, MotionFeatureConfig

# 创建配置
config = DatasetConfig(
    root_dir="./data",
    labels_file="labels.json",
    norm_logs_file="norm_logs.json",
    subset=['TYR', 'LJ', 'TX', 'WZT'],
    img_size=(32, 32),
    num_workers=4
)

# 配置运动特征
feature_config = MotionFeatureConfig(
    feature_names=['speed', 'acceleration', 'rotation_speed'],
    normalize=True
)

# 创建数据集和数据加载器
dataset = VRMotionSicknessDataset(config)
dataloader = get_vr_dataloader(
    dataset, 
    batch_size=16,
    feature_config=feature_config
)
```

## 数据加载器配置

### DatasetConfig 参数说明
- `root_dir` (str): 数据集根目录
- `labels_file` (str): 标签JSON文件路径
- `norm_logs_file` (str): 标准化日志JSON文件路径
- `subset` (Optional[List[str]]): 要包含的受试者ID列表
- `img_size` (Tuple[int, int]): 目标图像尺寸
- `num_workers` (int): 工作进程数量
- `cache_size` (int): LRU缓存大小（默认：512）
- `use_lmdb` (bool): 是否使用LMDB缓存（默认：False）
- `prefetch` (bool): 启用数据预取（默认：True）
- `prefetch_size` (int): 预取样本数量（默认：32）
- `max_prefetch_batches` (int): 最大预取批次数（默认：2）
- `prefetch_timeout` (int): 预取操作超时时间（默认：30秒）
- `min_free_memory_mb` (int): 保持的最小可用内存（默认：2048MB）
- `max_cache_memory_mb` (int): 缓存使用的最大内存（默认：8192MB）

### MotionFeatureConfig 参数说明
- `feature_names` (List[str]): 要包含的运动特征列表
- `normalize` (bool): 是否对特征进行标准化（默认：True）
- `padding_value` (float): 序列填充值（默认：0.0）
- `max_seq_length` (Optional[int]): 最大序列长度（默认：None）

## get_vr_dataloader 函数说明

```python
def get_vr_dataloader(
    dataset: VRMotionSicknessDataset,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    feature_config: Optional[MotionFeatureConfig] = None,
    use_prefetcher: bool = True
) -> Union[DataLoader, DataPrefetcher]
```

### 参数说明：
- `dataset`: VRMotionSicknessDataset实例
- `batch_size`: 每批次样本数量
- `num_workers`: 工作进程数量
- `pin_memory`: 是否在GPU训练时锁定内存
- `seed`: 随机种子，用于复现性
- `feature_config`: 运动特征配置
- `use_prefetcher`: 是否使用DataPrefetcher

### 返回值：
- PyTorch DataLoader或DataPrefetcher实例

## 批次数据格式
数据加载器生成的批次具有以下结构：
```python
{
    'frames_optical': torch.Tensor,    # 形状: [B, T, C, H, W]
    'frames_original': torch.Tensor,   # 形状: [B, T, C, H, W]
    'motion_features': torch.Tensor,   # 形状: [B, T, F]
    'labels': torch.Tensor,           # 形状: [B]
    'mask': torch.Tensor,             # 形状: [B, T]
    'motion_metadata': {
        'subject_ids': List[str],
        'slice_ids': List[str]
    },
    'eeg_data': torch.Tensor          # 形状: [B, C, T]
}
```
其中：
- B: 批次大小
- T: 序列长度
- C: 通道数
- H: 图像高度
- W: 图像宽度
- F: 运动特征数量

## 最佳实践

### 1. 内存管理
- 根据可用内存设置适当的`cache_size`
- 使用`dataset.cache.get_stats()`监控内存使用情况
- GPU训练时启用`pin_memory=True`

### 2. 性能优化
- 启用`use_prefetcher`加快数据加载
- 根据CPU核心数调整`num_workers`
- 使用`persistent_workers=True`避免工作进程初始化开销

### 3. 特征处理
- 在`MotionFeatureConfig`中配置相关的运动特征
- 启用标准化以获得稳定的训练
- 根据模型需求设置合适的`max_seq_length`

### 4. 错误处理
- 加载器会自动处理缺失的帧/数据
- 检查日志中关于缺失数据的警告
- 监控注意力掩码中的序列填充情况

## 训练循环示例
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 访问批次数据
        optical_frames = batch['frames_optical']
        original_frames = batch['frames_original']
        motion_features = batch['motion_features']
        labels = batch['labels']
        mask = batch['mask']
        
        # 你的训练逻辑
        ...
        
        # 定期监控缓存统计
        if iteration % 100 == 0:
            stats = dataset.cache.get_stats()
            print(f"缓存统计: {stats}")
```

## 注意事项
1. 数据预处理：
   - 确保所有输入数据都经过适当的标准化
   - 检查并处理异常值和缺失值
   - 保持数据格式的一致性

2. 性能监控：
   - 定期检查数据加载速度
   - 监控内存使用情况
   - 注意CPU和GPU利用率

3. 错误处理：
   - 妥善处理文件不存在的情况
   - 优雅处理损坏的数据
   - 记录并报告异常情况

4. 数据验证：
   - 检查批次数据的维度正确性
   - 验证特征值的范围
   - 确保标签的正确性