import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class BatchData:
    frames_optical: torch.Tensor  
    frames_original: torch.Tensor 
    motion_features: torch.Tensor 
    motion_metadata: Dict[str, List]
    labels: torch.Tensor  
    mask: torch.Tensor    

@dataclass
class MotionFeatureConfig:
    """运动特征配置"""
    feature_names: List[str] = ('speed', 'acceleration', 'rotation_speed')
    normalize: bool = True

@dataclass
class DatasetConfig:
    root_dir: str
    labels_file: str 
    norm_logs_file: str
    subset: Optional[List[str]] = None
    cache_size: int = 100
    max_cache_memory_mb: float = 1024  # 缓存可使用的最大内存(MB)
    min_free_memory_mb: float = 512    # 系统需要保持的最小空闲内存(MB)
    num_workers: int = 4
    prefetch: bool = True
    img_size: Tuple[int, int] = (224, 224)  # 添加图像尺寸配置
    use_lmdb: bool = False #是否使用LMDB
    lmdb_path: str = "frame_cache"
    prefetch_timeout: int = 300  # 预加载超时时间（秒）
    prefetch_size: int = 4      # 每批预加载的数量
    max_prefetch_batches: int = 5  # 最大预加载批次