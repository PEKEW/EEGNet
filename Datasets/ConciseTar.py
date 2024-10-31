#!/usr/bin/env python3
import os
import tarfile
import tempfile
from pathlib import Path
from collections import defaultdict
import shutil
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TarCompressor:
    def __init__(self, source_dir: str, output_dir: str, max_workers: int = 4):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.temp_dir = Path(tempfile.mkdtemp())
        self.lock = threading.Lock()
        
    def _get_subject_id(self, filename: str) -> str:
        """从文件名中提取subject_id"""
        return filename.split('_')[0]
    
    def _extract_tar(self, tar_path: Path, extract_path: Path):
        """提取单个tar文件"""
        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=extract_path)
        except Exception as e:
            logger.error(f"Error extracting {tar_path}: {e}")
            raise
            
    def _create_new_tar(self, files: list, output_path: Path):
        """创建新的tar文件"""
        try:
            with tarfile.open(output_path, 'w:gz') as tar:
                for file_path in files:
                    tar.add(file_path, arcname=file_path.name)
        except Exception as e:
            logger.error(f"Error creating {output_path}: {e}")
            raise
            
    def _process_subject(self, subject_id: str, tar_files: list):
        """处理单个subject的所有tar文件"""
        try:
            # 为每个subject创建临时目录
            subject_temp_dir = self.temp_dir / subject_id
            subject_temp_dir.mkdir(exist_ok=True)
            
            # 提取所有相关的tar文件
            for tar_file in tar_files:
                self._extract_tar(tar_file, subject_temp_dir)
            
            # 创建新的压缩tar
            output_path = self.output_dir / f"{subject_id}.tar.gz"
            self._create_new_tar(
                list(subject_temp_dir.glob('*')), 
                output_path
            )
            
            # 清理临时文件
            shutil.rmtree(subject_temp_dir)
            
            with self.lock:
                logger.info(f"Successfully processed subject {subject_id}")
                
        except Exception as e:
            logger.error(f"Error processing subject {subject_id}: {e}")
            raise
            
    def compress(self):
        """执行压缩过程"""
        try:
            # 确保输出目录存在
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 收集所有tar文件并按subject分组
            subject_files = defaultdict(list)
            tar_files = list(self.source_dir.glob('*.tar'))
            
            if not tar_files:
                raise FileNotFoundError(f"No tar files found in {self.source_dir}")
                
            for tar_file in tar_files:
                subject_id = self._get_subject_id(tar_file.stem)
                subject_files[subject_id].append(tar_file)
            
            logger.info(f"Found {len(tar_files)} tar files for {len(subject_files)} subjects")
            
            # 使用线程池处理每个subject
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 创建任务列表
                future_to_subject = {
                    executor.submit(self._process_subject, subject_id, files): subject_id
                    for subject_id, files in subject_files.items()
                }
                
                # 使用tqdm显示进度
                for future in tqdm(
                    future_to_subject,
                    total=len(future_to_subject),
                    desc="Processing subjects"
                ):
                    subject_id = future_to_subject[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Failed to process subject {subject_id}: {e}")
                        
        finally:
            # 清理所有临时文件
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                
        logger.info("Compression completed successfully")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compress multiple tar files by subject')
    parser.add_argument('source_dir', help='Directory containing source tar files')
    parser.add_argument('output_dir', help='Directory for output tar files')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    args = parser.parse_args()
    
    compressor = TarCompressor(args.source_dir, args.output_dir, args.workers)
    compressor.compress()

if __name__ == '__main__':
    main()
