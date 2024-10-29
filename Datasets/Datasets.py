# -- todo 实现基于内存限制的缓存策略，而不是固定数量
# -- todo 使用torch.utils.data.DataLoader的collate_fn进行批量预处理
# -- todo 使用@functools.lru_cache替代手动缓存实现
# -- todo 将部分预处理使用torchvision.transforms替代opencv操作
# -- todo 使用torch.multiprocessing替代threading
# -- todo 实现worker_init_fn 使用合适的 worker 数量
# todo 实现但没有测试 -- todo 将连续帧打包成单个文件(如.tar)，减少I/O操作
# todo EEGStream的加载过程
# todo code cleanup 把一些缓存相关的代码移到单独的工具文件
# todo 测试代码中的特征统计似乎不正确，需要检查
# todo GPU算力支持？考虑是否需要在这里实现
# todo 使用PyTorch的ImageFolder或WebDataset替代手动图片加载
# todo torch.utils.data.get_worker_info()在worker间共享缓存
# todo 把图像的resize放在处理tar的过程之前 做oneshot
from multiprocessing.managers import BaseManager
from torchvision import transforms
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, Queue
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List, Any, Union, NamedTuple, TypeVar, Generic
import lmdb
from dataclasses import dataclass
import mmap
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from contextlib import contextmanager
from torch.utils.data import get_worker_info
import random
import os
from torch import nn
from PIL import Image
import hashlib
from torch.nn.utils.rnn import pad_sequence
import io
import psutil
from threading import Lock
import sys
import gc
import threading
import time
T = TypeVar('T')


class WorkerSharedCache:
    """Worker间共享的缓存实现"""
    _shared_cache = {}  # 主进程缓存
    _worker_caches = {}  # worker进程缓存
    _lock = threading.Lock()
    
    def __init__(self, max_memory_mb: float = 1024, min_free_memory_mb: float = 512):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.min_free_memory = min_free_memory_mb * 1024 * 1024
        self.memory_usage = {}
        self.access_count = {}
        self.last_access = {}
        
        # 获取worker信息
        self.worker_info = get_worker_info()
        if self.worker_info is None:
            # 主进程
            self.cache = self._shared_cache
        else:
            # worker进程
            if self.worker_info.id not in self._worker_caches:
                self._worker_caches[self.worker_info.id] = {}
            self.cache = self._worker_caches[self.worker_info.id]
    
    @staticmethod
    def _get_tensor_size(tensor: torch.Tensor) -> int:
        """获取tensor的内存占用"""
        return tensor.element_size() * tensor.nelement()
    
    def _get_object_size(self, obj: Any) -> int:
        """获取对象的内存占用"""
        if isinstance(obj, torch.Tensor):
            return self._get_tensor_size(obj)
        elif isinstance(obj, (tuple, list)):
            return sum(self._get_object_size(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(self._get_object_size(v) for v in obj.values())
        return sys.getsizeof(obj)
    
    def _get_current_memory_usage(self) -> float:
        """获取当前内存使用量"""
        return sum(self.memory_usage.values())
    
    def _can_add_to_cache(self, size: int) -> bool:
        """检查是否可以添加新对象"""
        vm = psutil.virtual_memory()
        current_usage = self._get_current_memory_usage()
        return (current_usage + size <= self.max_memory and 
                vm.available - size >= self.min_free_memory)
    
    def _evict_entries(self, required_space: int) -> None:
        """驱逐缓存条目"""
        current_time = time.time()
        
        # 计算每个条目的得分
        scores = {}
        for key in list(self.cache.keys()):
            if key not in self.last_access:
                # 如果没有访问记录，直接移除
                self._remove_item(key)
                continue
                
            time_factor = 1.0 / (current_time - self.last_access[key] + 1)
            access_factor = self.access_count.get(key, 0)
            size_factor = 1.0 / (self.memory_usage.get(key, 1) + 1)
            
            scores[key] = (time_factor * access_factor * size_factor)
        
        # 按得分排序并驱逐
        sorted_items = sorted(scores.items(), key=lambda x: x[1])
        freed_space = 0
        
        for key, _ in sorted_items:
            if freed_space >= required_space:
                break
            freed_space += self.memory_usage.get(key, 0)
            self._remove_item(key)
    
    def _remove_item(self, key: str) -> None:
        """移除缓存项"""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.memory_usage:
                del self.memory_usage[key]
            if key in self.access_count:
                del self.access_count[key]
            if key in self.last_access:
                del self.last_access[key]
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        # 首先检查worker缓存
        if self.worker_info is not None and key in self.cache:
            value = self.cache[key]
        # 然后检查共享缓存
        elif key in self._shared_cache:
            value = self._shared_cache[key]
            # 如果在worker进程中，复制到worker缓存
            if self.worker_info is not None:
                self.cache[key] = value
        else:
            return None
            
        # 更新访问统计
        with self._lock:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            self.last_access[key] = time.time()
            
        return value
    
    def put(self, key: str, value: Any) -> bool:
        """添加项到缓存"""
        try:
            # 计算对象大小
            obj_size = self._get_object_size(value)
            
            # 如果对象太大，直接返回
            if obj_size > self.max_memory:
                return False
            
            # 检查是否需要驱逐
            if not self._can_add_to_cache(obj_size):
                self._evict_entries(obj_size)
                
                # 再次检查
                if not self._can_add_to_cache(obj_size):
                    return False
            
            # 更新缓存
            with self._lock:
                if self.worker_info is None:
                    # 主进程：放入共享缓存
                    self._shared_cache[key] = value
                else:
                    # worker进程：放入worker缓存
                    self.cache[key] = value
                
                self.memory_usage[key] = obj_size
                self.access_count[key] = 1
                self.last_access[key] = time.time()
            
            return True
            
        except Exception as e:
            print(f"Error putting item in cache: {e}")
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            if self.worker_info is None:
                self._shared_cache.clear()
            else:
                self.cache.clear()
            self.memory_usage.clear()
            self.access_count.clear()
            self.last_access.clear()
            gc.collect()
    
    def get_stats(self) -> Dict[str, float]:
        """获取缓存统计信息"""
        with self._lock:
            total_memory = self._get_current_memory_usage()
            vm = psutil.virtual_memory()
            
            stats = {
                'total_items': len(self.cache),
                'shared_items': len(self._shared_cache),
                'total_memory_mb': total_memory / (1024 * 1024),
                'memory_limit_mb': self.max_memory / (1024 * 1024),
                'available_memory_mb': vm.available / (1024 * 1024),
                'utilization': total_memory / self.max_memory if self.max_memory > 0 else 0
            }
            
            if self.worker_info is not None:
                stats['worker_id'] = self.worker_info.id
                stats['worker_items'] = len(self.cache)
            
            return stats

class ProcessSafeDict:
    """进程安全的字典实现"""
    def __init__(self):
        self._manager = mp.Manager()
        self._dict = self._manager.dict()
        self._lock = mp.Lock()
    
    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]
    
    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value
    
    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]
    
    def clear(self):
        with self._lock:
            self._dict.clear()
    
    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)
    
    def items(self):
        with self._lock:
            return list(self._dict.items())

class MPCompatibleMemoryCache:
    """支持多进程的内存受限缓存"""
    _instances = {}
    _lock = mp.Lock()
    
    @classmethod
    def get_instance(cls, name: str = "default", max_memory_mb: float = 1024, 
                    min_free_memory_mb: float = 512) -> 'MPCompatibleMemoryCache':
        """获取缓存实例（单例模式）"""
        with cls._lock:
            if name not in cls._instances:
                instance = cls(max_memory_mb, min_free_memory_mb)
                cls._instances[name] = instance
            return cls._instances[name]
    
    def __init__(self, max_memory_mb: float = 1024, min_free_memory_mb: float = 512):
        """
        Args:
            max_memory_mb: 缓存最大内存限制(MB)
            min_free_memory_mb: 系统最小剩余内存(MB)
        """
        self.max_memory = max_memory_mb * 1024 * 1024
        self.min_free_memory = min_free_memory_mb * 1024 * 1024
        
        # 使用进程安全的字典
        self._cache = ProcessSafeDict()
        self._memory_usage = ProcessSafeDict()
        self._access_count = ProcessSafeDict()
        self._last_access = ProcessSafeDict()
        
        # 清理线程
        self._cleanup_thread = None
        self._stop_cleanup = False
    
    def _get_object_size(self, obj: T) -> int:
        """获取对象占用的内存大小"""
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        return sys.getsizeof(obj)
    
    def _can_add_to_cache(self, size: int) -> bool:
        """检查是否可以添加新对象"""
        vm = psutil.virtual_memory()
        current_cache_size = sum(self._memory_usage.get(k, 0) for k in self._memory_usage._dict.keys())
        return (current_cache_size + size <= self.max_memory and 
                vm.available - size >= self.min_free_memory)
    
    def _cleanup_old_entries(self):
        """清理旧的缓存条目"""
        current_time = time.time()
        threshold = current_time - 3600  # 1小时未访问的条目
        
        for key, last_access in list(self._last_access.items()):
            if last_access < threshold:
                self.remove(key)
    
    def _start_cleanup_thread(self):
        """启动清理线程"""
        if self._cleanup_thread is None:
            def cleanup_loop():
                while not self._stop_cleanup:
                    self._cleanup_old_entries()
                    time.sleep(60)  # 每1钟清理一次
            
            self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
            self._cleanup_thread.start()
    
    def get(self, key: str) -> Optional[T]:
        """获取缓存项"""
        value = self._cache.get(key)
        if value is not None:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            self._last_access[key] = time.time()
        return value
    
    def put(self, key: str, value: T) -> bool:
        """添加项到缓存"""
        try:
            obj_size = self._get_object_size(value)
            
            if obj_size > self.max_memory:
                return False
                
            if not self._can_add_to_cache(obj_size):
                self._evict_entries(obj_size)
                
                if not self._can_add_to_cache(obj_size):
                    return False
            
            self._cache[key] = value
            self._memory_usage[key] = obj_size
            self._access_count[key] = 1
            self._last_access[key] = time.time()
            
            return True
        except Exception as e:
            print(f"Error putting item in cache: {e}")
            return False
    
    def remove(self, key: str) -> None:
        """移除缓存项"""
        try:
            del self._cache[key]
            del self._memory_usage[key]
            del self._access_count[key]
            del self._last_access[key]
        except KeyError:
            pass
    
    def _evict_entries(self, required_space: int) -> None:
        """驱逐缓存条目直到有足够空间"""
        items = [(k, self._access_count.get(k, 0) / (self._memory_usage.get(k, 1) + 1))
                for k in self._cache._dict.keys()]
        
        items.sort(key=lambda x: x[1])
        freed_space = 0
        
        for key, _ in items:
            if freed_space >= required_space:
                break
            freed_space += self._memory_usage.get(key, 0)
            self.remove(key)
    
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._memory_usage.clear()
        self._access_count.clear()
        self._last_access.clear()
        gc.collect()
    
    def get_stats(self) -> Dict[str, float]:
        """获取缓存统计信息"""
        total_memory = sum(self._memory_usage.get(k, 0) for k in self._memory_usage._dict.keys())
        vm = psutil.virtual_memory()
        
        return {
            'total_items': len(self._cache._dict),
            'total_memory_mb': total_memory / (1024 * 1024),
            'memory_limit_mb': self.max_memory / (1024 * 1024),
            'available_memory_mb': vm.available / (1024 * 1024),
            'utilization': total_memory / self.max_memory if self.max_memory > 0 else 0
        }


class MemoryBoundedCache(Generic[T]):
    """基于内存限制的缓存实现"""
    def __init__(self, max_memory_mb: float = 1024, min_free_memory_mb: float = 512):
        """
        Args:
            max_memory_mb: 缓存可使用的最大内存(MB)
            min_free_memory_mb: 系统需要保持的最小空闲内存(MB)
        """
        self.max_memory = max_memory_mb * 1024 * 1024  # 转换为字节
        self.min_free_memory = min_free_memory_mb * 1024 * 1024
        self.cache: Dict[str, T] = {}
        self.memory_usage: Dict[str, int] = {}
        self.access_count: Dict[str, int] = {}
        self.lock = Lock()
        
    def _get_object_size(self, obj: T) -> int:
        """获取对象的内存占用"""
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        return sys.getsizeof(obj)
    
    def _get_current_memory_usage(self) -> float:
        """获取当前系统内存使用情况"""
        process = psutil.Process()
        return process.memory_info().rss
    
    def _get_available_memory(self) -> float:
        """获取可用内存"""
        return psutil.virtual_memory().available
    
    def _can_add_to_cache(self, size: int) -> bool:
        """检查是否可以添加新对象到缓存"""
        current_cache_size = sum(self.memory_usage.values())
        available_memory = self._get_available_memory()
        
        return (current_cache_size + size <= self.max_memory and 
                available_memory - size >= self.min_free_memory)
    
    def _evict(self, required_space: int) -> None:
        """驱逐缓存项直到有足够空间"""
        if not self.cache:
            return
            
        # 按访问频率和大小计算得分
        scores = {
            key: self.access_count[key] / (self.memory_usage[key] + 1)
            for key in self.cache
        }
        
        # 按得分排序，驱逐得分最低的
        sorted_items = sorted(scores.items(), key=lambda x: x[1])
        
        freed_space = 0
        for key, _ in sorted_items:
            if freed_space >= required_space:
                break
            freed_space += self.memory_usage[key]
            del self.cache[key]
            del self.memory_usage[key]
            del self.access_count[key]
    
    def get(self, key: str) -> Optional[T]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                self.access_count[key] += 1
                return self.cache[key]
            return None
    
    def put(self, key: str, value: T) -> bool:
        """添加项到缓存"""
        with self.lock:
            # 计算对象大小
            obj_size = self._get_object_size(value)
            
            # 如果对象太大，直接返回
            if obj_size > self.max_memory:
                return False
            
            # 检查是否需要驱逐
            if not self._can_add_to_cache(obj_size):
                self._evict(obj_size)
                
                # 再次检查是否有足够空间
                if not self._can_add_to_cache(obj_size):
                    return False
            
            # 添加到缓存
            self.cache[key] = value
            self.memory_usage[key] = obj_size
            self.access_count[key] = 1
            return True
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.memory_usage.clear()
            self.access_count.clear()
            gc.collect()  # 强制垃圾回收
            
    def get_stats(self) -> Dict[str, float]:
        """获取缓存统计信息"""
        with self.lock:
            total_memory = sum(self.memory_usage.values())
            return {
                'total_items': len(self.cache),
                'total_memory_mb': total_memory / (1024 * 1024),
                'memory_limit_mb': self.max_memory / (1024 * 1024),
                'available_memory_mb': self._get_available_memory() / (1024 * 1024),
                'utilization': total_memory / self.max_memory if self.max_memory > 0 else 0
            }


class BatchItem(NamedTuple):
    """定义batch中单个样本的结构"""
    frames_optical: torch.Tensor
    frames_original: torch.Tensor
    motion_data: Dict[str, Any]
    label: torch.Tensor

class BatchData(NamedTuple):
    """定义整个batch的结构"""
    frames_optical: torch.Tensor  # [B, T, C, H, W]
    frames_original: torch.Tensor # [B, T, C, H, W]
    motion_features: torch.Tensor # [B, T, F]
    motion_metadata: Dict[str, List]
    labels: torch.Tensor  # [B]
    mask: torch.Tensor    # [B, T]

@dataclass
class MotionFeatureConfig:
    """运动特征配置"""
    feature_names: List[str] = ('speed', 'acceleration', 'rotation_speed')
    normalize: bool = True
    
class CollateProcessor:
    def __init__(self, feature_config: Optional[MotionFeatureConfig] = None):
        self.feature_config = feature_config or MotionFeatureConfig()
        self.feature_stats = {}
        
    def _extract_motion_features(self, motion_data: Dict) -> torch.Tensor:
        """从运动数据中提取特征"""
        features = []
        motion_features = motion_data['motion_features']
        
        for frame_idx in range(30):
            frame_key = f'frame_{frame_idx}'
            if frame_key not in motion_features:
                continue
                
            frame_data = motion_features[frame_key]
            frame_features = []
            
            for feature_name in self.feature_config.feature_names:
                try:
                    value = frame_data.get(feature_name, '0')
                    value = float(value.strip('()').split(',')[0]) if isinstance(value, str) else float(value)
                    frame_features.append(value)
                except (ValueError, IndexError):
                    frame_features.append(0.0)
                    
            features.append(frame_features)
            
        return torch.tensor(features, dtype=torch.float32)
    
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
        
        return BatchData(
            frames_optical=frames_optical_padded,
            frames_original=frames_original_padded,
            motion_features=motion_features_normalized,
            motion_metadata=motion_metadata,
            labels=torch.stack(labels),
            mask=mask
        )




class EnsureThreeChannels(nn.Module):
    """确保图像是三通道的自定义转换类"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


@dataclass
class DatasetConfig:
    root_dir: str
    labels_file: str 
    norm_logs_file: str
    subset: Optional[List[str]] = None
    # cache_size: int = 100
    max_cache_memory_mb: float = 1024  # 缓存可使用的最大内存(MB)
    min_free_memory_mb: float = 512    # 系统需要保持的最小空闲内存(MB)
    num_workers: int = 4
    prefetch: bool = True
    img_size: Tuple[int, int] = (224, 224)  # 添加图像尺寸配置
    use_lmdb: bool = False #是否使用LMDB
    lmdb_path: str = "frame_cache"

class FrameCache:
    """使用LMD实现缓存加速"""
    def __init__(self, path: str, map_size:int = int(1e12)):
        self.env = lmdb.open(
            path,
            map_size=map_size,
            readonly=False,
            meminit=False,
            map_async=True
        )
    """修改LMDB缓存以支持PIL Image"""
    def put(self, key: str, frame: np.ndarray):
        """存储图像到LMDB"""
        # 转换为PIL Image然后保存为bytes
        img = Image.fromarray(frame)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        
        with self.env.begin(write=True) as txn:
            txn.put(key.encode(), buffer.getvalue())
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """从LMDB读取图像"""
        with self.env.begin() as txn:
            buffer = txn.get(key.encode())
            if buffer is not None:
                # 从bytes重建PIL Image然后转换为numpy数组
                buffer = io.BytesIO(buffer)
                img = Image.open(buffer)
                return np.array(img)
            return None
    def close(self):
        self.env.close()
    
class VRMotionSicknessDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.root_dir = Path(config.root_dir)


        self._cache_size = 0

        # 定义图像转换流程
        # 这里可以添加自定义额外的数据增强
        self.transform = transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.ToTensor(),  # 自动将PIL Image转换为tensor并归一化到[0,1]
        ])

        # 定义光流图像的特殊转换流程（保持灰度信息）
        self.optical_transform = transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
            # lambda 不能序列化 需要使用显式声明
            EnsureThreeChannels()
        ])


        mp.set_start_method('spawn', force=True)

        # pool 不能序列化
        
        if config.use_lmdb:
            self.frame_cache = FrameCache(config.lmdb_path)
        else:
            self.frame_cache = None

        self.labels = self._load_json_mmap(config.labels_file)
        self.norm_logs = self._load_json_mmap(config.norm_logs_file)
        self.samples = self._build_samples(config.subset)
        

        manager = BaseManager()
        manager.start()
        self.cache = manager.MPCompatibleMemoryCache(
            max_memory_mb=config.max_cache_memory_mb,
            min_free_memory_mb=config.min_free_memory_mb
        )

        self.cache = WorkerSharedCache(
            max_memory_mb=config.max_cache_memory_mb,
            min_free_memory_mb=config.min_free_memory_mb
        )


        # self._cache = {}
        # self._cache_keys = []
        # self._cache_size = config.cache_size
        
        if config.prefetch:
            self._prefetch_data()

        # 设置缓存大小
        # self._configure_cache(config.cache_size)

    def _configure_cache(self, cache_size: int):
        """配置各种缓存的大小"""
        # 动态设置类方法的缓存大小
        self.__class__._cached_load_image = lru_cache(maxsize=cache_size)(self._load_image)
        self.__class__._cached_process_motion = lru_cache(maxsize=cache_size)(self._process_motion_features)
        self.__class__._cached_load_sequence = lru_cache(maxsize=cache_size)(self._load_frame_sequence)
    

    @staticmethod
    def _get_cache_key(*args, **kwargs) -> str:
        """生成缓存键"""
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = "|".join(key_parts)
        
        # 使用哈希来保证键的长度一致
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_json_mmap(self, file_path: str) -> Dict:
        with open(self.root_dir / file_path, 'r') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            return json.loads(mm.read().decode('utf-8'))

    @contextmanager
    def _get_pool(self):
        """上下文管理器处理进程池 
        不能把pool作为dataloader的属性
        因为dataloader会被序列化,
        而pool不能被序列化"""
        if self.config.num_workers > 0:
            with mp.Pool(processes=self.config.num_workers) as pool:
                yield pool
        else:
            yield None

    def _prefetch_data(self):
        """使用上下文管理器安全地预加载数据"""
        try:
            # 预取20%的数据，但不少于10个，不多于100个
            prefetch_size = max(10, min(int(len(self.samples) * 0.2), 100))
            self._cache_size = prefetch_size
            
            with self._get_pool() as pool:
                if pool is None:
                    return
                
                # 准备加载参数
                load_args = [
                    (
                        self.samples[idx]['subject_id'],
                        self.samples[idx]['slice_id'],
                        str(self.root_dir),
                        self.config.img_size
                    )
                    for idx in range(prefetch_size)
                ]
                
                # 异步加载
                results = pool.starmap_async(
                    self._load_sample_static,
                    load_args
                )
                
                try:
                    for idx, result in enumerate(results.get(timeout=30)):
                        self._update_cache(idx, result)
                except mp.TimeoutError:
                    print("Prefetch timeout, continuing with partial data")
                except Exception as e:
                    print(f"Error during prefetch: {e}")
                    
            print("Data prefetching completed")
            
        except Exception as e:
            print(f"Error in prefetch_data: {e}")
    

    @staticmethod
    def _load_sample_static(subject_id: str, slice_id: str, root_dir: str, img_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """静态方法用于多进程加载，避免序列化问题"""
        try:
            # 处理路径 这是一个typo导致的遗留问题 不过修改typo需要动脑子
            slice_id = slice_id[-1]
            frames_optical = []
            frames_original = []
            
            for frame_idx in range(29):
                optical_path = f'{root_dir}/sub_{subject_id}_sclice_{slice_id}_frame_{frame_idx}_optical.png'
                original_path = f'{root_dir}/sub_{subject_id}_sclice_{slice_id}_frame_{frame_idx}_original.png'
                
                optical_frame = VRMotionSicknessDataset._load_and_process_image(optical_path, img_size)
                original_frame = VRMotionSicknessDataset._load_and_process_image(original_path, img_size)
                
                frames_optical.append(optical_frame)
                frames_original.append(original_frame)
            
            frames_optical_tensor = torch.stack(frames_optical)
            frames_original_tensor = torch.stack(frames_original)
            
            return frames_optical_tensor, frames_original_tensor
            
        except Exception as e:
            raise RuntimeError(f"Error in load_sample_static: {e}")
        
    def _load_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
            顺序：光流图像序列，原始图像序列，运动特征，标签
        """
        try:
            sample = self.samples[idx]
            subject_id = sample['subject_id']
            slice_id = sample['slice_id']

            frames_optical, frames_original = self._load_frame_sequence(subject_id, slice_id)
            frames_optical_tensor = torch.FloatTensor(frames_optical)
            frames_original_tensor = torch.FloatTensor(frames_original)

            motion_features = self._process_motion_features(subject_id, slice_id)


            motion_data = {
                'subject_id': subject_id,
                'slice_id': slice_id,
                'motion_features': motion_features
            }

            label_tensor = torch.tensor(sample['label'], dtype=torch.float32)

            return frames_optical_tensor, frames_original_tensor, motion_data, label_tensor
        except Exception as e:
            raise RuntimeError(f"Error loading sample {idx}: {e}")



    def _validate_data(self):
        """验证数据完整性"""
        # 确保标签和日志数据匹配
        subjects_mismatch = set(self.labels.keys()) - set(self.norm_logs.keys())
        if subjects_mismatch:
            self.logger.warning(f"Mismatched subjects between labels and logs: {subjects_mismatch}")

    @lru_cache(maxsize=1024)
    def _process_motion_features(self, subject_id: str, slice_id: str) -> Dict:
        """处理运动特征，确保所有帧的特征都存在
        DataLoader会使用collate_fn来把多个样本合并成一个batch，这个函数会把不同样本的数据合并成一个字典
        如果字典的key不能一致就会抛出keyerror异常
        所以要把所有的确实的特征都填充上默认值
        """
        raw_features = self.norm_logs[subject_id][slice_id]
        default_feature = {
            "complete_sickness": False,
            "is_sickness": "0",
            "time_": 0.0,
            "pos": "(0.00,0.00,0.0)",
            "speed": "0",
            "acceleration": "0",
            "rotation_speed": "0"
        }
        
        return {
            f'frame_{i}': raw_features.get(f'frame_{i}', default_feature.copy())
            for i in range(30)
        }
    
    
    def _build_samples(self, subset: Optional[List[str]]) -> List[Dict]:
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


    @staticmethod
    @lru_cache(maxsize=128)
    def _load_json(file_path: Union[str, Path]) -> dict:
        """缓存JSON加载"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading JSON from {file_path}: {e}")


    def _preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """统一的图像预处理"""
        if frame is None:
            raise ValueError("Empty frame received")

        frame = cv2.resize(frame, self.config.img_size)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    @lru_cache(maxsize=None)  # None表示无限缓存
    def _load_image(self, img_path: str) -> torch.Tensor:
        """缓存图像加载结果"""
        try:
            image = Image.open(img_path).convert('RGB')
            if 'optical' in img_path:
                return self.optical_transform(image)
            return self.transform(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

    @staticmethod
    def _load_and_process_image(img_path: str, size: Tuple[int, int]) -> torch.Tensor:
        """静态方法用于多进程加载"""
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        
        optical_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
        ])
        
        try:
            image = Image.open(img_path).convert('RGB')
            if 'optical' in img_path:
                return optical_transform(image)
            return transform(image)
        except Exception as e:
            raise RuntimeError(f"Error processing image {img_path}: {e}")

    @lru_cache(maxsize=512)
    def _load_frame_sequence(self, subject_id: str, slice_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载图像序列"""
        slice_id = slice_id[-1]
        frames_optical = []
        frames_original = []
        
        try:
            for frame_idx in range(29):
                optical_path = str(self.root_dir / f'sub_{subject_id}_sclice_{slice_id}_frame_{frame_idx}_optical.png')
                original_path = str(self.root_dir / f'sub_{subject_id}_sclice_{slice_id}_frame_{frame_idx}_original.png')
                
                # 使用新的加载方法
                optical_frame = self._load_image(optical_path)
                original_frame = self._load_image(original_path)
                
                frames_optical.append(optical_frame)
                frames_original.append(original_frame)
                
            return torch.stack(frames_optical), torch.stack(frames_original)
            
        except Exception as e:
            raise RuntimeError(f"Error loading frame sequence: {e}")
    
    def clear_cache(self):
        """清除所有缓存"""
        self._load_image.cache_clear()
        self._process_motion_features.cache_clear()
        self._load_frame_sequence.cache_clear()
        self._load_json.cache_clear()

    def __del__(self):
        """清理资源时清除缓存"""
        self.clear_cache()


    # def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], torch.Tensor]:
    #     """获取数据项，使用缓存的方法"""
    #     sample = self.samples[idx]

    #     subject_id = sample['subject_id']
    #     slice_id = sample['slice_id']
        
    #     # 使用缓存的方法加载数据
    #     frames_optical, frames_original = self._load_frame_sequence(subject_id, slice_id)
    #     motion_features = self._process_motion_features(subject_id, slice_id)
        
    #     motion_data = {
    #         'subject_id': subject_id,
    #         'slice_id': slice_id,
    #         'motion_features': motion_features
    #     }
        
    #     label_tensor = torch.tensor(sample['label'], dtype=torch.float32)
        
    #     return frames_optical, frames_original, motion_data, label_tensor


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], torch.Tensor]:
        """获取数据项，使用缓存的方法"""
        sample = self.samples[idx]
        subject_id = sample['subject_id']
        slice_id = sample['slice_id']
        
        # 为不同类型的数据使用不同的缓存键
        frames_cache_key = f"frames_{subject_id}_{slice_id}"
        motion_cache_key = f"motion_{subject_id}_{slice_id}"
        
        # 尝试从缓存获取帧数据
        cached_frames = self.cache.get(frames_cache_key)
        if cached_frames is not None:
            frames_optical, frames_original = cached_frames
        else:
            frames_optical, frames_original = self._load_frame_sequence(subject_id, slice_id)
            self.cache.put(frames_cache_key, (frames_optical, frames_original))
        
        # 尝试从缓存获取运动特征
        cached_motion = self.cache.get(motion_cache_key)
        if cached_motion is not None:
            motion_features = cached_motion
        else:
            motion_features = self._process_motion_features(subject_id, slice_id)
            self.cache.put(motion_cache_key, motion_features)
        
        # 构建motion_data字典
        motion_data = {
            'subject_id': subject_id,
            'slice_id': slice_id,
            'motion_features': motion_features
        }
        
        # 标签不需要缓存，直接创建张量
        label_tensor = torch.tensor(sample['label'], dtype=torch.float32)
        
        return frames_optical, frames_original, motion_data, label_tensor


    def _update_cache(self, idx: int, data: tuple):
        """Update LRU cache"""
        if idx in self._cache:
            self._cache_keys.remove(idx)
        elif len(self._cache) >= self._cache_size:
            # Remove oldest item
            oldest = self._cache_keys.pop(0)
            del self._cache[oldest]
            
        self._cache[idx] = data
        self._cache_keys.append(idx)
    
    def __len__(self):
        return len(self.samples)

# def collate_fn(batch):
#     """Custom collate function to handle the motion features dictionary"""
#     frames_optical, frames_original, motion_data, labels = zip(*batch)
    
#     return (
#         torch.stack(frames_optical),
#         torch.stack(frames_original),
#         motion_data,  # Keep as tuple of dicts
#         torch.stack(labels)
#     )


def worker_init_fn(worker_id: int) -> None:
    """
    初始化DataLoader的每个worker进程
    
    Args:
        worker_id: DataLoader分配的worker ID
    """
    worker_info = get_worker_info()
    
    if worker_info is None:
        return
        
    base_seed = worker_info.seed
    worker_seed = (base_seed + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)
    
    torch.manual_seed(worker_seed)
    
    # 确保CUDA操作是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    dataset = worker_info.dataset
    
    worker_id = worker_info.id
    
    if hasattr(dataset, 'config') and dataset.config.use_lmdb:
        worker_cache_dir = f"{dataset.config.lmdb_path}_worker_{worker_id}"
        os.makedirs(worker_cache_dir, exist_ok=True)
        
        # 为每个worker创建独立的LMDB环境
        if hasattr(dataset, 'frame_cache'):
            dataset.frame_cache = FrameCache(worker_cache_dir)


def get_vr_dataloader(
    dataset: VRMotionSicknessDataset,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    feature_config: Optional[MotionFeatureConfig] = None
) -> DataLoader:
    """
    创建VR数据加载器
    
    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        num_workers: worker进程数
        pin_memory: 是否将数据固定在内存中
        seed: 随机种子
    """

    # 设置全局随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 设置Generator用于DataLoader的随机打乱
    g = torch.Generator()
    g.manual_seed(seed)
    

    collate_processor = CollateProcessor(feature_config)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True,
        collate_fn=collate_processor,
        worker_init_fn=worker_init_fn,
        generator=g,
        prefetch_factor=2  # 预取因子，每个worker预加载的batch数
    )


if __name__ == '__main__':
    BaseManager.register('MPCompatibleMemoryCache', MPCompatibleMemoryCache)
    config = DatasetConfig(
        root_dir="./data",
        labels_file="labels.json",
        norm_logs_file="norm_logs.json",
        subset=['TYR'],
        img_size=(32, 32),
        num_workers=4
    )

    feature_config = MotionFeatureConfig(
        feature_names=['speed', 'acceleration', 'rotation_speed'],
        normalize=True
    )
    dataset = VRMotionSicknessDataset(config)
    dataloader = get_vr_dataloader(dataset, feature_config=feature_config)

    print(f"Dataset size: {len(dataset)}")
    
    for idx, batch in enumerate(dataloader):


        if idx % 100 == 0:
            stats = dataset.cache.get_stats()
            print(f"Cache stats: {stats}")

        print(f"\nBatch {idx} loaded successfully")
        print("Shapes:")
        print(f"- Optical frames: {batch.frames_optical.shape}")
        print(f"- Original frames: {batch.frames_original.shape}")
        print(f"- Motion features: {batch.motion_features.shape}")
        print(f"- Labels: {batch.labels.shape}")
        print(f"- Attention mask: {batch.mask.shape}")
        
        print("\nMetadata:")
        print(f"- Number of samples: {len(batch.motion_metadata['subject_ids'])}")
        print(f"- Subject IDs: {batch.motion_metadata['subject_ids'][:3]}...")
        
        print("\nTensor Properties:")
        print(f"- Device: {batch.frames_optical.device}")
        print(f"- Optical frames range: [{batch.frames_optical.min():.3f}, {batch.frames_optical.max():.3f}]")
        print(f"- Motion features mean: {batch.motion_features.mean():.3f}")
        print(f"- Valid sequences: {batch.mask.sum(dim=1).tolist()}")
        
        # 详细的特征统计
        if hasattr(dataloader.collate_fn, 'feature_stats'):
            print("\nFeature Statistics:")
            stats = dataloader.collate_fn.feature_stats
            if stats:
                print(f"- Mean: {stats['mean'].tolist()}")
                print(f"- Std: {stats['std'].tolist()}")
        
        break