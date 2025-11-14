#!/usr/bin/env python3
"""
AI Cloud Storage 测试脚本
测试云存储系统的各项功能
"""

import os
import sys
import tempfile
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import CloudStorageManager

def test_cloud_storage():
    """测试云存储功能"""
    print("🧪 开始测试 AI Cloud Storage...")
    
    # 初始化存储管理器
    storage_manager = CloudStorageManager()
    print("✅ 存储管理器初始化成功")
    
    # 测试文件类型检测
    test_files = [
        "test.xlsx",
        "document.pdf", 
        "image.png",
        "data.csv",
        "video.mp4"
    ]
    
    print("\n📁 测试文件类型检测:")
    for filename in test_files:
        file_type = storage_manager.get_file_type(filename)
        icon = storage_manager.get_file_icon(file_type)
        print(f"  {filename} -> {file_type} {icon}")
    
    # 测试文件大小格式化
    print("\n📏 测试文件大小格式化:")
    test_sizes = [0, 1024, 1024*1024, 1024*1024*1024]
    for size in test_sizes:
        formatted = storage_manager.format_file_size(size)
        print(f"  {size} bytes -> {formatted}")
    
    # 测试文件夹创建
    print("\n📁 测试文件夹创建:")
    folder_result = storage_manager.create_folder("测试文件夹")
    if folder_result["success"]:
        print(f"  ✅ 文件夹创建成功: ID {folder_result['folder_id']}")
    else:
        print(f"  ❌ 文件夹创建失败: {folder_result['error']}")
    
    # 测试文件搜索
    print("\n🔍 测试文件搜索:")
    search_results = storage_manager.search_files("test")
    print(f"  搜索 'test' 找到 {len(search_results)} 个文件")
    
    # 测试上传进度
    print("\n🔄 测试上传进度:")
    progress_list = storage_manager.get_upload_progress()
    print(f"  当前有 {len(progress_list)} 个未完成的上传")
    
    print("\n✅ 所有测试完成!")

if __name__ == "__main__":
    test_cloud_storage()
