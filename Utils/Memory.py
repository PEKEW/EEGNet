import threading  # tag
import torch
from torch.utils.data import get_worker_info
import psutil
import sys
import gc
import multiprocessing as mp
from typing import Any, Optional, Dict, TypeVar, Generic
from PIL import Image
import io
import numpy as np
from torch.multiprocessing import Lock
import random
import os
import lmdb
import time
from collections import OrderedDict

T = TypeVar('T')


def check_memory_pressure(min_free_memory_mb) -> bool:
    """检查系统内存压力"""
    try:
        memory = psutil.virtual_memory()
        return memory.available / 1024 / 1024 < min_free_memory_mb
    except:
        return False

class LRUCache:
    """简单的LRUCache实现
    """
    def __init__(self, capacity: int = 100):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def put(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.put(key, value)

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
