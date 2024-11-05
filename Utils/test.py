import os
import sys


def test_working_directory():
    # 获取当前工作目录
    current_dir = os.getcwd()
    # 获取文件所在目录
    file_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取Python路径
    python_path = sys.path
    
    print("\n=== 工作目录测试 ===")
    print(f"当前工作目录: {current_dir}")
    print(f"文件所在目录: {file_dir}")
    print(f"Python路径: {python_path}")
    print("\n=== models目录测试 ===")
    # 检查models目录是否存在
    models_dir = os.path.join(current_dir, './models')
    if os.path.exists(models_dir):
        print(f"models目录存在: {models_dir}")
        # 列出models目录中的文件
        print("models目录中的文件:")
        for file in os.listdir(models_dir):
            print(f"  - {file}")
    else:
        print("警告: models目录不存在!")