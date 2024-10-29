import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import lmdb
from dataclasses import dataclass
import mmap
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# todo 使用PyTorch的ImageFolder或WebDataset替代手动图片加载
# todo torch.utils.data.get_worker_info()在worker间共享缓存
# todo 将连续帧打包成单个文件(如.tar)，减少I/O操作
# todo 实现基于内存限制的缓存策略，而不是固定数量
# todo 使用@functools.lru_cache替代手动缓存实现
# todo 使用torch.utils.data.DataLoader的collate_fn进行批量预处理
# todo 将部分预处理移至GPU(如resize、归一化) 使用torchvision.transforms替代opencv操作，更好的GPU支持
# todo 使用torch.multiprocessing替代threading
# todo 实现worker_init_fn正确初始化每个worker

@dataclass
class DatasetConfig:
    root_dir: str
    labels_file: str 
    norm_logs_file: str
    subset: Optional[List[str]] = None
    cache_size: int = 100
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
    def put(self, key: str, frame: np.ndarray):
        """帧数据存储LMDB
        """
        success, buffer = cv2.imencode('.png', frame)
        if success:
            with self.env.begin(write=True) as txn:
                txn.put(key.encode(), buffer.tobytes())
    
    def get(self, key: str) -> Optional[np.ndarray]:
        "从LMDB读"
        with self.env.begin() as txn:
            buffer = txn.get(key.encode())
            if buffer is not None:
                frame_buffer = np.frombuffer(buffer, dtype=np.uint8)
                frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                return frame
            return None
    def close(self):
        self.env.close()
    
class VRMotionSicknessDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.root_dir = Path(config.root_dir)
        
        # Initialize LMDB if needed
        if config.use_lmdb:
            self.frame_cache = FrameCache(config.lmdb_path)
        else:
            self.frame_cache = None
            
        # Load data using memory mapping
        self.labels = self._load_json_mmap(config.labels_file)
        self.norm_logs = self._load_json_mmap(config.norm_logs_file)
        self.samples = self._build_samples(config.subset)
        
        # Simple cache implementation
        self._cache = {}
        self._cache_keys = []
        self._cache_size = config.cache_size
        
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
            
        self.samples = self._build_samples(config.subset)
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        
        if config.prefetch:
            self._prefetch_data()

    def _load_json_mmap(self, file_path: str) -> Dict:
        with open(self.root_dir / file_path, 'r') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            return json.loads(mm.read().decode('utf-8'))


    def _prefetch_data(self):
        """预加载数据"""
        prefetch_size = min(self._cache_size, len(self.samples))
        try:
            with ThreadPoolExecutor() as executor:
                executor.map(self._load_sample, range(prefetch_size))
            print("数据预加载完成")
        except Exception as e:
            raise RuntimeError(f"Error prefetching data: {e}")
    
    def _validate_paths(self):
        """验证所有必要的路径是否存在"""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")
        
        required_files = [
            self.root_dir / self.config.labels_file,
            self.root_dir / self.config.norm_logs_file
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")

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


    @lru_cache(maxsize=128)
    def _load_json(self, file_path: str) -> Dict:
        """使用LRU缓存加载JSON文件"""
        try:
            with open(self.root_dir / file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load {file_path}: {e}")

    def _preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """统一的图像预处理"""
        if frame is None:
            raise ValueError("Empty frame received")
            
        # 调整图像大小
        frame = cv2.resize(frame, self.config.img_size)
        
        # 颜色空间转换和归一化
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        
        return frame


    def _load_image(self, img_path: Path) -> torch.Tensor:
        if self.frame_cache is not None:
            cached_frame = self.frame_cache.get(str(img_path))
            if cached_frame is not None:
                frame = cached_frame
            else:
                frame = cv2.imread(str(img_path))
                if frame is not None:
                    self.frame_cache.put(str(img_path), frame)
        else:
            frame = cv2.imread(str(img_path))
            
        if frame is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        # Convert to RGB and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.config.img_size)
        
        # Convert to tensor
        return torch.FloatTensor(frame).permute(2, 0, 1) / 255.0

    @staticmethod
    def _load_and_process_image(img_path: str, size: Tuple[int, int]) -> torch.Tensor:
        """Static method for image loading to ensure pickle compatibility"""
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)
        
        return torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
    
    def _load_frame_sequence(self, subject_id: str, slice_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        slice_id = slice_id[-1]
        frames_optical = []
        frames_original = []
        
        for frame_idx in range(29):
            optical_path = str(self.root_dir / f'sub_{subject_id}_sclice_{slice_id}_frame_{frame_idx}_optical.png')
            original_path = str(self.root_dir / f'sub_{subject_id}_sclice_{slice_id}_frame_{frame_idx}_original.png')
            
            frames_optical.append(self._load_and_process_image(optical_path, self.config.img_size))
            frames_original.append(self._load_and_process_image(original_path, self.config.img_size))
                
        return torch.stack(frames_optical), torch.stack(frames_original)
    

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], torch.Tensor]:
        # Check cache
        if idx in self._cache:
            return self._cache[idx]

        sample = self.samples[idx]
        subject_id = sample['subject_id']
        slice_id = sample['slice_id']

        # Load data
        frames_optical, frames_original = self._load_frame_sequence(subject_id, slice_id)
        motion_features = self._process_motion_features(subject_id, slice_id)
        
        motion_data = {
            'subject_id': subject_id,
            'slice_id': slice_id,
            'motion_features': motion_features
        }
        
        label_tensor = torch.tensor(sample['label'], dtype=torch.float32)
        
        # Update cache
        data = (frames_optical, frames_original, motion_data, label_tensor)
        self._update_cache(idx, data)
            
        return data

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

    # def _manage_cache(self, idx: int, data: tuple):
    #     """缓存管理"""
    #     try:
    #         if len(self.cache) >= self.cache_size:
    #             self.cache.pop(self.cache_queue.pop(0))
    #         self.cache[idx] = data
    #         self.cache_queue.append(idx)
    #     except Exception as e:
    #         raise RuntimeError(f"Error managing cache for index {idx}: {e}")

    # def clear_cache(self):
    #     """安全地清除缓存"""
    #     try:
    #         self.cache.clear()
    #         self.cache_queue.clear()
    #         self.logger.info("Cache cleared successfully")
    #     except Exception as e:
    #         self.logger.error(f"Failed to clear cache: {e}")



def collate_fn(batch):
    """Custom collate function to handle the motion features dictionary"""
    frames_optical, frames_original, motion_data, labels = zip(*batch)
    
    return (
        torch.stack(frames_optical),
        torch.stack(frames_original),
        motion_data,  # Keep as tuple of dicts
        torch.stack(labels)
    )

def get_vr_dataloader(
    dataset: VRMotionSicknessDataset,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True,
        collate_fn=collate_fn
    )


if __name__ == '__main__':
    config = DatasetConfig(
        root_dir="./data",
        labels_file="labels.json",
        norm_logs_file="norm_logs.json",
        subset=['TYR'],
        img_size=(32, 32),
        num_workers=4
    )

    dataset = VRMotionSicknessDataset(config)
    dataloader = get_vr_dataloader(dataset)

    print(f"Dataset size: {len(dataset)}")
    
    for idx, (frames_optical, frames_original, motion_data, labels) in enumerate(dataloader):
        print(f"Batch {idx} loaded successfully")
        print(f"Optical frames shape: {frames_optical.shape}")
        print(f"Original frames shape: {frames_original.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Number of motion data samples: {len(motion_data)}")
        break