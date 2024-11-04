# VR眩晕数据集使用文档

## 概述
VRSicknessDataset 是一个基于 PyTorch Dataset 的实现，专门用于处理和分析 VR 导致的眩晕数据。该数据集包含多种数据模态，包括视频帧、脑电图(EEG)信号、运动数据以及眩晕标签。

## 数据集结构

### 必需的目录结构
```
root_dir/
├── frame_archives/    # 包含视频帧的tar文件
│   └── {subject_id}_combined.tar
├── EEGData/          # 包含EEG数据的tar文件
│   └── {subject_id}.tar
├── norm_logs.json    # 运动数据日志
└── labels.json       # 眩晕标签
```

### 数据格式说明

#### 1. 视频帧数据
- 存储在名为 `{subject_id}_combined.tar` 的tar档案中
- 包含两种类型的帧：
  - 原始帧（`original.png`）
  - 光流帧（`optical.png`）
- 命名规则：`sub_{subject_id}_sclice_{slice_id}_frame_{frame_id}_{type}.png`

#### 2. 脑电图(EEG)数据
- 存储在名为 `{subject_id}.tar` 的tar档案中
- 使用EEGLAB格式（.set文件）
- 命名规则：`{subject_id}_slice_{slice_id}.set`

#### 3. 运动数据（norm_logs.json）
数据结构：
```json
{
    "受试者ID": {
        "片段X": {
            "帧Y": {
                "time_": float,        # 时间戳
                "speed": float,        # 速度
                "acceleration": float, # 加速度
                "rotation_speed": float, # 旋转速度
                "is_sickness": int,    # 是否出现眩晕
                "complete_sickness": int, # 是否完全眩晕
                "pos": "(x,y,z)"      # 位置坐标
            }
        }
    }
}
```

#### 4. 标签数据（labels.json）
数据结构：
```json
{
    "受试者ID": {
        "片段X": float  # 眩晕程度得分
    }
}
```

## 主要特性

### 数据加载机制
- 视频帧和运动数据采用惰性加载
- EEG数据预先加载到内存以提高效率
- 自动验证所有模态数据的完整性
- 内置支持视频帧数据转换

### 数据同步
- 确保所有模态的时间对齐
- 验证各数据源之间的对应关系
- 维护帧序列的一致性

### 性能优化
- 高效的tar文件处理
- 内存优化的EEG数据加载
- 支持CUDA GPU加速
- 支持多进程处理，使用spawn方法

## 使用方法

### 基本用法
```python
# 创建数据集实例
dataset = VRSicknessDataset(root_dir="数据路径")

# 创建数据加载器
dataloader = get_dataloader(
    root_dir="数据路径",
    batch_size=32,            # 批次大小
    num_workers=4,            # 工作进程数
    sequence_length=None,     # 使用批次中最长序列长度
    padding_mode='repeat'     # 填充模式
)
```

### 返回数据格式
每个数据项是一个包含以下内容的字典：
```python
{
    'sub_id': str,              # 受试者ID
    'slice_id': str,            # 片段ID
    'optical_frames': Tensor,   # 形状: [帧数, 通道数, 高度, 宽度]
    'original_frames': Tensor,  # 形状: [帧数, 通道数, 高度, 宽度]
    'eeg': Tensor,             # 形状: [通道数, 时间点数]
    'motion': Tensor,          # 形状: [帧数, 特征数]
    'label': Tensor            # 形状: [1]
}
```

### 数据统计
对每个样本，数据集提供以下统计信息：
- 均值、标准差、最小值、最大值，包括：
  - 原始视频帧
  - 光流帧
  - EEG信号
  - 运动数据

## 性能注意事项

### 内存管理
- EEG数据预加载到内存
- 视频帧和运动数据按需加载
- 自动清理临时文件
- 支持CUDA固定内存，加速GPU传输

### 多进程处理
- 使用'spawn'启动方法确保兼容性
- 支持多工作进程
- 支持持久化工作进程提高效率
- 可配置预取因子

### 错误处理
- 健壮的数据缺失和损坏检查
- 详细的错误信息便于调试
- 优雅处理模态缺失情况

## 依赖项
- torch：深度学习框架
- torchvision：图像处理工具
- PIL：图像处理库
- mne：脑电图处理库
- json：JSON数据处理
- tarfile：tar文件处理
- warnings：警告处理
- os, sys：系统操作
- re：正则表达式
- pathlib：路径处理

## 最佳实践建议
1. 根据可用CPU核心数设置工作进程数
2. 有GPU时启用CUDA加速
3. 根据可用内存调整批次大小
4. 考虑序列长度以平衡批次
5. 监控大数据集的内存使用情况

## 已知限制
1. 需要足够的RAM预加载EEG数据
2. 使用/tmp目录存储临时文件
3. 依赖统一的命名约定
4. 要求所有模态数据都存在才视为有效样本