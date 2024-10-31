from multiprocessing import Manager # tag
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from Utils.Memory import MPCompatibleMemoryCache, WorkerSharedCache, worker_init_fn, FrameCache, LRUCache, check_memory_pressure
from multiprocessing.managers import BaseManager
from torchvision import transforms
import torch.multiprocessing as mp
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List, Any, Union, Callable
import mmap
from functools import lru_cache
from contextlib import contextmanager
from PIL import Image
import hashlib
from DatasetsUtils import EnsureThreeChannels, DataPrefetcher, DeviceCollator, CollateProcessor
import tarfile
from DataClass import  DatasetConfig, MotionFeatureConfig
from io import BytesIO
import mne
import tempfile
import warnings

class VRFrameDataset(Dataset):
    def __init__(
        self,
        root: str,
        frame_type: str,
        transform: Optional[Callable] = None,
        frame_count: int = 29,
        cache_size: int = 512  # 缓存大小
    ):
        self.root = root
        self.frame_type = frame_type
        self.transform = transform
        self.frame_count = frame_count
        self.samples = []
        self.tar_handles = {}
        self.frame_cache = LRUCache(cache_size)  # 添加LRU缓存
        self.transform_cache = Manager().dict()  # 使用共享字典
        self.eeg_cache = {}  # 添加EEG缓存
    def _load_eeg(self, tar_path: str, subject_id: str, slice_id: str) -> torch.Tensor:
        """从tar文件加载EEG数据"""
        cache_key = f"{tar_path}:{subject_id}:{slice_id}_eeg"
        
        if cache_key in self.eeg_cache:
            return self.eeg_cache[cache_key]
                
        eeg_tar_path = os.path.join(self.root, 'EEGData', f"{subject_id}.tar")
        
        try:
            if not os.path.exists(eeg_tar_path):
                print(f"Warning: EEG tar file not found: {eeg_tar_path}")
                return torch.zeros((64, 1000))
                
            if eeg_tar_path not in self.tar_handles:
                try:
                    self.tar_handles[eeg_tar_path] = tarfile.open(eeg_tar_path, 'r')
                except Exception as e:
                    print(f"Error opening tar file {eeg_tar_path}: {e}")
                    return torch.zeros((64, 1000))
                    
            tar = self.tar_handles[eeg_tar_path]
            
            eeg_filename = f"{subject_id}_slice_{slice_id}.set"
            try:
                member = tar.getmember(eeg_filename)
            except KeyError:
                available_files = [
                    m.name for m in tar.getmembers() 
                    if m.name.startswith(f"{subject_id}_slice_")
                ][:5]
                print(f"Warning: EEG file {eeg_filename} not found in {eeg_tar_path}")
                print(f"Available similar files: {available_files}...")
                return torch.zeros((64, 1000))
                
            try:
                # 创建临时目录
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 提取.set和.fdt文件（如果存在）
                    temp_set_path = os.path.join(temp_dir, eeg_filename)
                    tar.extract(member, temp_dir)
                    
                    # 读取.set文件数据并转换为tensor
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        eeg_data = mne.io.read_raw_eeglab(temp_set_path, preload=True)
                    eeg_tensor = torch.from_numpy(eeg_data.get_data())
                    
                    self.eeg_cache[cache_key] = eeg_tensor
                    return eeg_tensor
                
            except Exception as e:
                print(f"Error processing EEG file {eeg_filename}: {e}")
                return torch.zeros((64, 1000))
                
        except Exception as e:
            print(f"Unexpected error loading EEG data: {e}")
            return torch.zeros((64, 1000))

    def _cached_transform(self, img_bytes: bytes, transform_name: str) -> torch.Tensor:
        """缓存基于图像字节和转换名称的转换结果"""
        cache_key = f"{hash(img_bytes)}_{transform_name}"
        if cache_key in self.transform_cache:
            return self.transform_cache[cache_key]

        img = Image.open(BytesIO(img_bytes))

        if transform_name == "optical":
            transformed_img = VRMotionSicknessDataset.optical_transform(img)
        else:
            transformed_img = VRMotionSicknessDataset.transform(img)

        self.transform_cache[cache_key] = transformed_img
        return transformed_img
    
    def _load_frame(self, tar_path: str, subject_id: str, slice_id: str, frame_name: str) -> torch.Tensor:
        """从tar文件中加载单帧"""
        cache_key = f"{tar_path}:{subject_id}:{slice_id}:{frame_name}"
        
        if cache_key in self.frame_cache:
            return self.frame_cache[cache_key]
        
        if tar_path not in self.tar_handles:
            self.tar_handles[tar_path] = tarfile.open(tar_path, 'r')
            
        tar = self.tar_handles[tar_path]

        try:
            # 构建tar文件中的完整路径
            internal_path = f"{subject_id}/{frame_name}"
            member = tar.getmember(internal_path)
            f = tar.extractfile(member)
            img_bytes = f.read()
            
            transformed_frame = self._cached_transform(img_bytes, self.frame_type)
            self.frame_cache[cache_key] = transformed_frame
            return transformed_frame
        except KeyError:
            available_files = [
                m.name for m in tar.getmembers() 
                if m.name.startswith(f"{subject_id}/") 
                and m.name.endswith(f"_{self.frame_type}.png")
            ][:5]
            raise FileNotFoundError(
                f"Frame {internal_path} not found in {tar_path}. "
                f"Available similar files: {available_files}..."
            )
    def __getitem__(self, index: Union[int, Tuple[str, str]]) -> torch.Tensor:
        if isinstance(index, tuple):
            subject_id, slice_id = index
        else:
            subject_id, slice_id = self.samples[index]
            
        slice_id = slice_id[-1]
        frames = []
        
        # 使用合并后的tar文件路径
        tar_path = os.path.join(self.root, 'frame_archives', f"{subject_id}_combined.tar")
        
        for i in range(self.frame_count):
            frame_name = f"sub_{subject_id}_sclice_{slice_id}_frame_{i}_{self.frame_type}.png"
            frame = self._load_frame(tar_path, subject_id, slice_id, frame_name)
            frames.append(frame)

        # 加载EEG数据
        eeg_data = self._load_eeg(tar_path, subject_id, slice_id)
        
        return {
            'frames': torch.stack(frames),
            'eeg_data': eeg_data,
            'metadata': {
                'subject_id': subject_id,
                'slice_id': slice_id,
                'frame_type': self.frame_type
            }
    }
    
    def __del__(self):
        """清理打开的tar文件"""
        for tar in self.tar_handles.values():
            tar.close()
            
    def add_sequence(self, subject_id: str, slice_id: str):
        """添加一个序列到数据集"""
        self.samples.append((subject_id, slice_id))

class VRMotionSicknessDataset(Dataset):
    # 定义图像转换流程
    # 这里可以添加自定义额外的数据增强
    transform = transforms.Compose([
        transforms.ToTensor(),  # 自动将PIL Image转换为tensor并归一化到[0,1]
    ])

    # 定义光流图像的特殊转换流程（保持灰度信息）
    optical_transform = transforms.Compose([
        transforms.ToTensor(),
        EnsureThreeChannels()
    ])
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.root_dir = Path(config.root_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.optical_folder = VRFrameDataset(
            root=str(self.root_dir),
            frame_type='optical',
            transform=self.optical_transform
        )
        
        self.original_folder = VRFrameDataset(
            root=str(self.root_dir),
            frame_type='original',
            transform=self.transform
        )
        mp.set_start_method('spawn', force=True)

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

         # 为数据集添加样本
        for sample in self.samples:
            self.optical_folder.add_sequence(sample['subject_id'], sample['slice_id'])
            self.original_folder.add_sequence(sample['subject_id'], sample['slice_id'])

        self._cache = {}
        self._cache_keys = []
        self._cache_size = config.cache_size
        
        if config.prefetch:
            self._prefetch_data()


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
        """优化的数据预加载函数"""
        try:
            if not self.config.prefetch:
                return

            # 计算预加载总量
            total_prefetch = min(
                self.config.prefetch_size * self.config.max_prefetch_batches,
                len(self.samples)
            )
            print(f"Starting data prefetch for {total_prefetch} samples...")
            with self._get_pool() as pool:
                if pool is None:
                    return
                # 分批预加载
                for batch_start in range(0, total_prefetch, self.config.prefetch_size):
                    batch_end = min(batch_start + self.config.prefetch_size, total_prefetch)
                    # 准备这一批的加载参数
                    load_args = [
                        (
                            self.samples[idx]['subject_id'],
                            self.samples[idx]['slice_id'],
                            str(self.root_dir),
                            self.config.img_size
                        )
                        for idx in range(batch_start, batch_end)
                    ]
                    
                    # 异步加载当前批次
                    results = pool.starmap_async(
                        self._load_sample_static,
                        load_args
                    )
                    try:
                        # 为每个批次设置超时
                        batch_results = results.get(timeout=self.config.prefetch_timeout)
                        for idx, result in enumerate(batch_results):
                            self._update_cache(batch_start + idx, result)
                        
                        print(f"Successfully prefetched batch {batch_start//self.config.prefetch_size + 1}")
                        
                    except mp.TimeoutError:
                        print(f"Batch {batch_start//self.config.prefetch_size + 1} prefetch timeout, continuing...")
                        continue
                    except Exception as e:
                        print(f"Error in batch {batch_start//self.config.prefetch_size + 1}: {e}")
                        continue
                    
                    # 检查内存使用情况
                    if check_memory_pressure(self.config.min_free_memory_mb):
                        print("Memory pressure detected, stopping prefetch")
                        break
            print(f"Prefetch completed. Cached {len(self._cache)} samples")
            
        except Exception as e:
            print(f"Error in prefetch_data: {e}")
            print("Continuing without full prefetch...")


    @staticmethod
    def _load_sample_static(subject_id: str, slice_id: str, root_dir: str, img_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """优化的静态加载方法"""
        try:
            slice_id = slice_id[-1]
            frames_optical = []
            frames_original = []
            
            # 打开tar文件
            tar_path = os.path.join(root_dir, 'frame_archives', f"{subject_id}_combined.tar")
            with tarfile.open(tar_path, 'r') as tar:
                for frame_idx in range(29):
                    # 构建文件路径
                    optical_name = f"{subject_id}/sub_{subject_id}_sclice_{slice_id}_frame_{frame_idx}_optical.png"
                    original_name = f"{subject_id}/sub_{subject_id}_sclice_{slice_id}_frame_{frame_idx}_original.png"
                    
                    try:
                        # 加载光流图像
                        optical_member = tar.getmember(optical_name)
                        optical_file = tar.extractfile(optical_member)
                        optical_image = Image.open(BytesIO(optical_file.read())).convert('RGB')
                        optical_tensor = VRMotionSicknessDataset._process_image(optical_image, True)
                        frames_optical.append(optical_tensor)
                        
                        # 加载原始图像
                        original_member = tar.getmember(original_name)
                        original_file = tar.extractfile(original_member)
                        original_image = Image.open(BytesIO(original_file.read())).convert('RGB')
                        original_tensor = VRMotionSicknessDataset._process_image(original_image, False)
                        frames_original.append(original_tensor)
                        
                    except KeyError as e:
                        print(f"Warning: Missing frame {frame_idx} for {subject_id}_{slice_id}")
                        frames_optical.append(torch.zeros(3, *img_size))
                        frames_original.append(torch.zeros(3, *img_size))
            
            # 加载EEG数据
            eeg_tar_path = os.path.join(root_dir, 'EEGData', f"{subject_id}.tar")
            try:
                with tarfile.open(eeg_tar_path, 'r') as tar:
                    eeg_filename = f"{subject_id}_slice_{slice_id}.set"
                    
                    # 创建临时目录
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # 提取.set文件
                        member = tar.getmember(eeg_filename)
                        temp_set_path = os.path.join(temp_dir, eeg_filename)
                        tar.extract(member, temp_dir)
                        
                        # 读取EEG数据
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            eeg_data = mne.io.read_raw_eeglab(temp_set_path, preload=True)
                        eeg_tensor = torch.from_numpy(eeg_data.get_data())
                        
            except Exception as e:
                print(f"Warning: Error loading EEG data for {subject_id}_{slice_id}: {e}")
                eeg_tensor = torch.zeros((64, 1000))  # 根据实际EEG数据维度调整
            
            return {
                'frames_optical': torch.stack(frames_optical),
                'frames_original': torch.stack(frames_original),
                'eeg_data': eeg_tensor
            }
            
        except Exception as e:
            print(f"Error loading sample {subject_id}_{slice_id}: {e}")
            raise
        
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

    @lru_cache(maxsize=1024)
    def _process_motion_features(self, subject_id: str, slice_id: str) -> Dict:
        """处理运动特征，将数值数据转换为tensor并移动到正确的设备"""
        raw_features = self.norm_logs[subject_id][slice_id]
        default_feature = {
            "complete_sickness": False,
            "is_sickness": "0",
            "time_": torch.tensor(0.0, device=self.device),
            "pos": torch.tensor([0.0, 0.0, 0.0], device=self.device),
            "speed": torch.tensor(0.0, device=self.device),
            "acceleration": torch.tensor(0.0, device=self.device),
            "rotation_speed": torch.tensor(0.0, device=self.device)
        }
        
        processed_features = {}
        for i in range(30):
            frame_key = f'frame_{i}'
            if frame_key in raw_features:
                raw_frame = raw_features[frame_key]
                processed_frame = {
                    "complete_sickness": raw_frame.get("complete_sickness", False),
                    "is_sickness": raw_frame.get("is_sickness", "0"),
                    "time_": torch.tensor(float(raw_frame.get("time_", 0.0)), device=self.device),
                    "pos": self._parse_position(raw_frame.get("pos", "(0.00,0.00,0.0)"), self.device),
                    "speed": torch.tensor(float(raw_frame.get("speed", "0")), device=self.device),
                    "acceleration": torch.tensor(float(raw_frame.get("acceleration", "0")), device=self.device),
                    "rotation_speed": torch.tensor(float(raw_frame.get("rotation_speed", "0")), device=self.device)
                }
            else:
                processed_frame = default_feature.copy()
            
            processed_features[frame_key] = processed_frame
        
        return processed_features

    def _parse_position(self, pos_str: str, device: str) -> torch.Tensor:
        """解析位置字符串为tensor"""
        try:
            # 去除括号并分割坐标
            coords = pos_str.strip('()').split(',')
            # 转换为浮点数并创建tensor
            return torch.tensor([float(x) for x in coords], device=device)
        except (ValueError, IndexError):
            return torch.tensor([0.0, 0.0, 0.0], device=device)

    
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
    def _process_image(img: Image.Image, is_optical: bool) -> torch.Tensor:
        """优化的图像处理方法"""
        if is_optical:
            return VRMotionSicknessDataset.optical_transform(img)
        return VRMotionSicknessDataset.transform(img)

    @staticmethod
    def _load_and_process_image(img_path: str, size: Tuple[int, int]) -> torch.Tensor:
        """静态方法用于多进程加载"""
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        optical_transform = transforms.Compose([
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
    
    def _load_frame_sequence(self, subject_id: str, slice_id: str) -> Dict[str, torch.Tensor]:
        """使用ImageFolder加载图像序列和EEG数据"""
        # 从缓存获取
        cache_key = f"frames_{subject_id}_{slice_id}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data is not None:
            frames_optical, frames_original, eeg_data = cached_data
            return {
                'frames_optical': frames_optical,
                'frames_original': frames_original,
                'eeg_data': eeg_data
            }
        
        # 使用ImageFolder加载序列
        optical_data = self.optical_folder[(subject_id, slice_id)]
        original_data = self.original_folder[(subject_id, slice_id)]
        
        if optical_data is None or original_data is None:
            raise RuntimeError(f"Missing data for subject {subject_id}, slice {slice_id}")
        
        # 准备数据
        frames_data = (
            optical_data['frames'].to(self.device),
            original_data['frames'].to(self.device),
            optical_data['eeg_data'].to(self.device)
        )
        
        # 存入缓存的是元组格式
        self.cache.put(cache_key, frames_data)
        
        # 返回字典格式
        return {
            'frames_optical': frames_data[0],
            'frames_original': frames_data[1],
            'eeg_data': frames_data[2]
        }
    


    def clear_cache(self):
        """清除所有缓存"""
        self._load_image.cache_clear()
        self._process_motion_features.cache_clear()
        self._load_frame_sequence.cache_clear()
        self._load_json.cache_clear()

    def __del__(self):
        """清理资源时清除缓存"""
        self.clear_cache()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], torch.Tensor]:
        """获取数据项"""
        sample = self.samples[idx]
        subject_id = sample['subject_id']
        slice_id = sample['slice_id']
        
        # 加载帧序列
        # frames_optical, frames_original = self._load_frame_sequence(subject_id, slice_id)

        # 加载帧序列和EEG数据
        sequence_data = self._load_frame_sequence(subject_id, slice_id)
        
        # 获取运动特征
        motion_cache_key = f"motion_{subject_id}_{slice_id}"
        motion_features = self.cache.get(motion_cache_key)
        
        if motion_features is None:
            motion_features = self._process_motion_features(subject_id, slice_id)
            self.cache.put(motion_cache_key, motion_features)
        
        motion_data = {
            'subject_id': subject_id,
            'slice_id': slice_id,
            'motion_features': motion_features
        }

        label_tensor = torch.tensor(sample['label'], dtype=torch.float32, device=self.device)
        
        # return frames_optical, frames_original, motion_data, label_tensor   

        return {
            'frames_optical': sequence_data['frames_optical'],
            'frames_original': sequence_data['frames_original'],
            'motion_data': motion_data,
            'label': label_tensor,
            'eeg_data': sequence_data['eeg_data']
        }


    def _update_cache(self, idx: int, data: Dict[str, torch.Tensor]):
        """Update LRU cache"""
        if idx in self._cache:
            self._cache_keys.remove(idx)
        elif len(self._cache) >= self._cache_size:
            # Remove oldest item
            oldest = self._cache_keys.pop(0)
            del self._cache[oldest]
            
        # 转换为元组格式存储
        cache_tuple = (
            data['frames_optical'],
            data['frames_original'],
            data['eeg_data']
        )
        self._cache[idx] = cache_tuple
        self._cache_keys.append(idx)
        
    def __len__(self):
        return len(self.samples)
    


def get_vr_dataloader(
    dataset: VRMotionSicknessDataset,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    feature_config: Optional[MotionFeatureConfig] = None,
    use_prefetcher: bool = True
) -> Union[DataLoader, DataPrefetcher]:
    """创建优化的VR数据加载器"""
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    collate_processor = CollateProcessor(feature_config)
    device_collator = DeviceCollator(collate_processor, dataset.device)
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=True,
        collate_fn=device_collator,
        worker_init_fn=worker_init_fn,
        generator=g,
        prefetch_factor=2
    )
    
    return DataPrefetcher(loader, dataset.device) if use_prefetcher else loader


if __name__ == '__main__':
    BaseManager.register('MPCompatibleMemoryCache', MPCompatibleMemoryCache)
    config = DatasetConfig(
        root_dir="./data",
        labels_file="labels.json",
        norm_logs_file="norm_logs.json",
        subset=['TYR', 'LJ', 'TX', 'WZT'],
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