import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from multiprocessing import Manager # tag
from Utils.Memory import MPCompatibleMemoryCache, WorkerSharedCache, worker_init_fn, FrameCache, LRUCache, check_memory_pressure
from multiprocessing.managers import BaseManager
from torchvision import transforms
import torch.multiprocessing as mp
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Dict, Tuple, Optional, List, Any, Union, Callable
import mmap
from functools import lru_cache
from contextlib import contextmanager
from PIL import Image
import hashlib
from DatasetsUtils import EnsureThreeChannels, DataPrefetcher, DeviceCollator, CollateProcessor, BatchData
import tarfile
from DataClass import  DatasetConfig, MotionFeatureConfig
from io import BytesIO
import mne
import tempfile
import warnings

FPS = 30-1

class VRFrameDataset(Dataset):
    channel_nums = 30 # EEG通道数 32原始 - 2参考
    srate = 250  # EEG采样率
    def __init__(
        self,
        root: str,
        frame_type: str,
        transform: Optional[Callable] = None,
        frame_count: int = FPS,
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
                print(f"Warning: EEG tar file not found: {eeg_tar_path}"
                      ", using the zero tensor as a placeholder")
                return torch.zeros((self.channel_nums, self.srate))
                
            if eeg_tar_path not in self.tar_handles:
                try:
                    self.tar_handles[eeg_tar_path] = tarfile.open(eeg_tar_path, 'r')
                except Exception as e:
                    print(f"Error opening tar file {eeg_tar_path}: {e}"
                          ", using the zero tensor as a placeholder")
                    return torch.zeros((self.channel_nums, self.srate))
            tar = self.tar_handles[eeg_tar_path]
            eeg_filename = f"{subject_id}_slice_{slice_id}.set"
            try:
                member = tar.getmember(eeg_filename)
            except KeyError:
                available_files = [
                    m.name for m in tar.getmembers() 
                    if m.name.startswith(f"{subject_id}_slice_")
                ][:5]
                print(f"Warning: EEG file {eeg_filename} not found in {eeg_tar_path}"
                f"Available similar files: {available_files}"
                ", using the zero tensor as a placeholder")
                return torch.zeros((self.channel_nums, self.srate))
            try:
                # 创建临时目录 把tar object提取到临时目录
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_set_path = os.path.join(temp_dir, eeg_filename)
                    tar.extract(member, temp_dir)
                    # 读取.set文件数据并转换为tensor
                    # 这里因为切片只处理的关键数据 会遇到一些警告 直接忽略
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        eeg_data = mne.io.read_raw_eeglab(temp_set_path, preload=True)
                    eeg_tensor = torch.from_numpy(eeg_data.get_data())
                    self.eeg_cache[cache_key] = eeg_tensor
                    return eeg_tensor
                
            except Exception as e:
                print(f"Error processing EEG file {eeg_filename}: {e}"
                      ", using the zero tensor as a placeholder")
                return torch.zeros((self.channel_nums, self.srate))
                
        except Exception as e:
            print(f"Unexpected error loading EEG data: {e}"
                  ", using the zero tensor as a placeholder")
            return torch.zeros((self.channel_nums, self.srate))

    def _cached_transform(self, img_bytes: bytes, transform_name: str) -> torch.Tensor:
        """缓存基于图像字节和转换名称的转换结果"""
        cache_key = f"{hash(img_bytes)}_{transform_name}"
        if cache_key in self.transform_cache:
            return self.transform_cache[cache_key]
        img = Image.open(BytesIO(img_bytes))

        transformed_img = VRMotionSicknessDataset.transform(img) \
        if transform_name == "original" else \
            VRMotionSicknessDataset.optical_transform(img)

        self.transform_cache[cache_key] = transformed_img
        return transformed_img
    
    def _load_frame(self, tar_path: str, subject_id: str, slice_id: str, frame_name: str) -> torch.Tensor:
        """从tar文件中加载单帧
        这里存在两层tar
        """
        cache_key = f"{tar_path}:{subject_id}:{slice_id}:{frame_name}"
        if cache_key in self.frame_cache:
            return self.frame_cache[cache_key]
        if tar_path not in self.tar_handles:
            self.tar_handles[tar_path] = tarfile.open(tar_path, 'r')
        tar = self.tar_handles[tar_path]

        try:
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

        # 因为文件命名错误 这里需要单独处理slice_id   
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
        EnsureThreeChannels() # 确保是3通道的 需要序列化 但是不能使用lambda 
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

        self.frame_cache = FrameCache(config.lmdb_path) \
        if config.use_lmdb else None

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
        # 绑定到类而不是实例 可以在所有实例之间共享
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
        """数据预加载"""
        try:
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
                for frame_idx in range(FPS):
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
                        print(f"Warning: Missing frame {frame_idx} for {subject_id}_{slice_id}"
                              ", using zero tensor as a placeholder")
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
                eeg_tensor = torch.zeros((VRFrameDataset.channel_nums, VRFrameDataset.srate))
            
            return {
                'frames_optical': torch.stack(frames_optical),
                'frames_original': torch.stack(frames_original),
                'eeg_data': eeg_tensor
            }
            
        except Exception as e:
            print(f"Error loading sample {subject_id}_{slice_id}: {e}")
            raise
        

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
        for i in range(FPS):
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
                # print(f"motion frame {frame_key} not found in {subject_id}_{slice_id}, using default features")
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
        except (ValueError, IndexError) as e:
            print(f"Error parsing position string {pos_str}: {e}")
            return torch.tensor([0.0, 0.0, 0.0], device=device)

    
    def _build_samples(self, subset: Optional[List[str]]) -> List[Dict]:
        """
        只添加info 不实际添加样本
        """
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
        try:
            # 清理基本缓存
            if hasattr(self, 'frame_cache') and self.frame_cache is not None:
                self.frame_cache = {}
                
            # 安全清理transform缓存    
            if hasattr(self, 'optical_folder') and hasattr(self.optical_folder, 'transform_cache'):
                self.optical_folder.transform_cache.clear()
                
            if hasattr(self, 'original_folder') and hasattr(self.original_folder, 'transform_cache'):    
                self.original_folder.transform_cache.clear()
                
            # 清理内部缓存
            if hasattr(self, '_cache'):
                self._cache.clear()
                
            if hasattr(self, '_cache_keys'):    
                self._cache_keys.clear()
                
            # 安全清理共享缓存
            if hasattr(self, 'cache'):
                try:
                    self.cache.clear()
                except (FileNotFoundError, ConnectionError, AttributeError):
                    pass  # 忽略多进程管理器已关闭的错误
                    
        except Exception as e:
            print(f"Warning: Error during cache clearing: {e}")

    def __del__(self):
        """析构函数"""
        try:
            self.clear_cache()
        except Exception as e:
            print(f"Warning: Error during object destruction: {e}")
        finally:
            # 确保关闭所有打开的资源
            if hasattr(self, 'optical_folder'):
                try:
                    del self.optical_folder
                except:
                    pass
            if hasattr(self, 'original_folder'):    
                try:
                    del self.original_folder
                except:
                    pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], torch.Tensor]:
        """获取数据项"""
        sample = self.samples[idx]
        subject_id = sample['subject_id']
        slice_id = sample['slice_id']

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


def test_dataset_properties(dataset, num_samples=2):
    """
    测试数据集的tensor属性
    Args:
        dataset: VRMotionSicknessDataset实例
        num_samples: 要测试的样本数量
    """
    print("="*50)
    print("Testing Dataset Properties")
    print(f"Dataset Length: {len(dataset)}")
    print("="*50)
    
    for i in range(min(num_samples, len(dataset))):
        print(f"\nSample {i+1}:")
        print("-"*30)
        
        sample = dataset[i]
        
        # 检查光流帧
        frames_optical = sample['frames_optical']
        print("\nOptical Frames:")
        print(f"Shape: {frames_optical.shape}")
        print(f"Type: {frames_optical.dtype}")
        print(f"Device: {frames_optical.device}")
        print(f"Value range: [{frames_optical.min():.2f}, {frames_optical.max():.2f}]")
        
        # 检查原始帧
        frames_original = sample['frames_original']
        print("\nOriginal Frames:")
        print(f"Shape: {frames_original.shape}")
        print(f"Type: {frames_original.dtype}")
        print(f"Device: {frames_original.device}")
        print(f"Value range: [{frames_original.min():.2f}, {frames_original.max():.2f}]")
        
        # 检查EEG数据
        eeg_data = sample['eeg_data']
        print("\nEEG Data:")
        print(f"Shape: {eeg_data.shape}")
        print(f"Type: {eeg_data.dtype}")
        print(f"Device: {eeg_data.device}")
        print(f"Value range: [{eeg_data.min():.2f}, {eeg_data.max():.2f}]")
        
        # 检查标签
        label = sample['label']
        print("\nLabel:")
        print(f"Shape: {label.shape}")
        print(f"Type: {label.dtype}")
        print(f"Device: {label.device}")
        print(f"Value: {label.item()}")
        
        # 检查运动特征
        motion_data = sample['motion_data']
        print("\nMotion Features:")
        print(f"Subject ID: {motion_data['subject_id']}")
        print(f"Slice ID: {motion_data['slice_id']}")
        
        # 打印第一帧的运动特征示例
        first_frame = motion_data['motion_features']['frame_0']
        print("\nFirst Frame Motion Features:")
        for key, value in first_frame.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}:")
                print(f"  Shape: {value.shape}")
                print(f"  Type: {value.dtype}")
                print(f"  Device: {value.device}")
                print(f"  Value: {value}")
        
        print("="*50)

def debug_data_pipeline(dataset, dataloader, num_samples=2):
    """详细调试数据集和数据加载器的数据流转过程"""
    print("="*50)
    print("1. Dataset Single Sample Check")
    print("="*50)
    
    # 1. 检查原始数据集样本
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i} from dataset:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}:")
                print(f"  Shape: {value.shape}")
                print(f"  Range: [{value.min():.3f}, {value.max():.3f}]")
                print(f"  Mean: {value.mean():.3f}")
                print(f"  All zeros: {(value == 0).all().item()}")
                
    # 2. 检查collate过程
    print("\n" + "="*50)
    print("2. Collate Function Check")
    print("="*50)
    
    batch_samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]
    if hasattr(dataloader, 'collate_fn'):
        collated = dataloader.collate_fn(batch_samples)
        print("\nAfter collate_fn:")
        if isinstance(collated, BatchData):
            for field_name, value in collated.__dict__.items():
                if isinstance(value, torch.Tensor):
                    print(f"{field_name}:")
                    print(f"  Shape: {value.shape}")
                    print(f"  Range: [{value.min():.3f}, {value.max():.3f}]")
                    print(f"  Mean: {value.mean():.3f}")
                    print(f"  All zeros: {(value == 0).all().item()}")
    
    # 3. 检查DataLoader的第一个batch
    print("\n" + "="*50)
    print("3. DataLoader First Batch Check")
    print("="*50)
    
    try:
        first_batch = next(iter(dataloader))
        print("\nFirst batch from DataLoader:")
        if isinstance(first_batch, BatchData):
            for field_name, value in first_batch.__dict__.items():
                if isinstance(value, torch.Tensor):
                    print(f"{field_name}:")
                    print(f"  Shape: {value.shape}")
                    print(f"  Range: [{value.min():.3f}, {value.max():.3f}]")
                    print(f"  Mean: {value.mean():.3f}")
                    print(f"  All zeros: {(value == 0).all().item()}")
                    print(f"  Device: {value.device}")
    except Exception as e:
        print(f"Error in DataLoader: {e}")

# 添加动作特征处理的调试函数
def debug_motion_features(processor, motion_data):
    """调试动作特征处理过程"""
    print("="*50)
    print("Motion Features Processing Debug")
    print("="*50)
    
    # 1. 检查原始特征提取
    raw_features = processor._extract_motion_features(motion_data)
    print("\nRaw extracted features:")
    print(f"Shape: {raw_features.shape}")
    print(f"Range: [{raw_features.min():.3f}, {raw_features.max():.3f}]")
    print(f"Mean: {raw_features.mean():.3f}")
    print(f"All zeros: {(raw_features == 0).all().item()}")
    
    # 2. 检查特征归一化
    if processor.feature_config.normalize:
        normalized = processor._normalize_features(raw_features.unsqueeze(0))
        print("\nAfter normalization:")
        print(f"Shape: {normalized.shape}")
        print(f"Range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        print(f"Mean: {normalized.mean():.3f}")
        print(f"Stats mean: {processor.feature_stats.get('mean', None)}")
        print(f"Stats std: {processor.feature_stats.get('std', None)}")
        
    # 3. 检查原始motion_features的内容
    print("\nRaw motion_features content check:")
    frame_data = motion_data['motion_features'].get('frame_0', {})
    print("First frame features:")
    for feat_name in processor.feature_config.feature_names:
        if feat_name in frame_data:
            value = frame_data[feat_name]
            if isinstance(value, torch.Tensor):
                print(f"{feat_name}: {value.item():.3f}")
            else:
                print(f"{feat_name}: {value}")


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

    # test_dataset_properties(dataset, num_samples=2)

    dataloader = get_vr_dataloader(dataset, feature_config=feature_config, num_workers=4)
    debug_data_pipeline(dataset, dataloader)
    # 检查动作特征处理
    sample = dataset[0]
    processor = CollateProcessor(feature_config)
    debug_motion_features(processor, sample['motion_data'])