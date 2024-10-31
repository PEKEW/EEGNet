import os
import re
import tarfile
from collections import defaultdict

def pack_set_files(directory):
    # 编译正则表达式模式
    pattern = re.compile(r'([A-Z]+)_slice_\d+\.set$')
    
    # 用字典来组织文件
    file_groups = defaultdict(list)
    
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # 提取前缀(XXX部分)
            prefix = match.group(1)
            # 将文件添加到对应的组
            file_groups[prefix].append(filename)
    
    # 为每个前缀创建tar文件
    for prefix, files in file_groups.items():
        tar_filename = os.path.join(directory, f"{prefix}.tar")
        
        # 创建tar文件
        with tarfile.open(tar_filename, "w") as tar:
            for file in files:
                # 添加文件到tar
                file_path = os.path.join(directory, file)
                tar.add(file_path, arcname=file)
        
        print(f"Created {tar_filename} with {len(files)} files")

# 使用示例
if __name__ == "__main__":
    # 替换为你的目标目录路径
    directory = "."
    pack_set_files(directory)