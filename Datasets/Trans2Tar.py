# todo resize to 256*256

import os
from pathlib import Path
from tqdm import tqdm
import tarfile
import shutil
import argparse
import concurrent.futures
from typing import List, Dict, Set
import logging

class DatasetConverter:
    """将散落的图像文件转换为tar格式的数据集"""
    
    def __init__(self, source_dir: str, target_dir: str, num_workers: int = 4):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.num_workers = num_workers
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dataset_conversion.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _get_sequence_groups(self) -> Dict[str, List[Path]]:
        """将图像文件按序列分组"""
        groups = {}
        for img_path in self.source_dir.glob("*.png"):
            # 解析文件名 "sub_XXX_sclice_Y_frame_Z_(optical|original).png"
            parts = img_path.stem.split('_')
            subject_id = parts[1]
            slice_id = parts[3]
            group_key = f"{subject_id}_{slice_id}"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(img_path)
            
        return groups
    
    def _create_tar_archive(self, group_key: str, file_paths: List[Path]) -> bool:
        """为一个序列创建tar文件"""
        tar_path = self.target_dir / f"{group_key}.tar"
        try:
            with tarfile.open(tar_path, "w") as tar:
                for file_path in sorted(file_paths):  # 确保文件顺序一致
                    tar.add(file_path, arcname=file_path.name)
            return True
        except Exception as e:
            self.logger.error(f"Error creating tar for {group_key}: {e}")
            return False
    
    def _verify_tar_archive(self, tar_path: Path, expected_files: Set[str]) -> bool:
        """验证tar文件的完整性"""
        try:
            with tarfile.open(tar_path, "r") as tar:
                actual_files = set(m.name for m in tar.getmembers())
                return actual_files == expected_files
        except Exception as e:
            self.logger.error(f"Error verifying tar {tar_path}: {e}")
            return False
            
    def convert(self):
        """执行数据集转换"""
        self.logger.info("Starting dataset conversion")
        
        # 确保目标目录存在
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取序列分组
        groups = self._get_sequence_groups()
        self.logger.info(f"Found {len(groups)} sequences to process")
        
        # 记录处理结果
        results = {
            'success': 0,
            'failed': 0,
            'verified': 0
        }
        
        # 使用进度条显示转换进度
        with tqdm(total=len(groups), desc="Converting sequences") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # 提交所有转换任务
                future_to_group = {
                    executor.submit(self._create_tar_archive, group_key, files): group_key 
                    for group_key, files in groups.items()
                }
                
                # 处理完成的任务
                for future in concurrent.futures.as_completed(future_to_group):
                    group_key = future_to_group[future]
                    try:
                        success = future.result()
                        if success:
                            # 验证创建的tar文件
                            tar_path = self.target_dir / f"{group_key}.tar"
                            expected_files = {p.name for p in groups[group_key]}
                            if self._verify_tar_archive(tar_path, expected_files):
                                results['verified'] += 1
                                results['success'] += 1
                            else:
                                results['failed'] += 1
                                # 删除验证失败的tar文件
                                tar_path.unlink(missing_ok=True)
                        else:
                            results['failed'] += 1
                    except Exception as e:
                        self.logger.error(f"Error processing {group_key}: {e}")
                        results['failed'] += 1
                    
                    pbar.update(1)
        
        # 输出转换结果
        self.logger.info(f"""
        Dataset conversion completed:
        - Successfully converted: {results['success']}
        - Successfully verified: {results['verified']}
        - Failed: {results['failed']}
        """)

def main():
    parser = argparse.ArgumentParser(description="Convert dataset to tar format")
    parser.add_argument("--source", type=str, 
                    default="./data",  # 添加默认值
                    help="Source directory containing image files")
    parser.add_argument("--target", type=str, 
                    default="./data/frame_archives",  # 添加默认值
                    help="Target directory for tar files")
    parser.add_argument("--workers", type=int, default=4, 
                    help="Number of worker threads")
    
    args = parser.parse_args()
    
    # 验证源目录存在
    if not os.path.exists(args.source):
        print(f"Error: Source directory '{args.source}' does not exist")
        print("Please specify the correct source directory using --source")
        return
    
    # 创建目标目录
    os.makedirs(args.target, exist_ok=True)
    
    print(f"Converting files from {args.source} to {args.target}")
    print(f"Using {args.workers} workers")
    
    try:
        converter = DatasetConverter(
            source_dir=args.source,
            target_dir=args.target,
            num_workers=args.workers
        )
        converter.convert()
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    main()