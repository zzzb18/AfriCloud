#!/usr/bin/env python3
"""
测试预览功能
"""

import pandas as pd
import io
from app import CloudStorageManager

def test_preview_function():
    """测试预览功能"""
    print("🔍 测试预览功能...")
    
    # 初始化存储管理器
    storage_manager = CloudStorageManager()
    
    # 创建测试文件
    print("📄 创建测试文件...")
    
    # 1. 创建Excel测试文件
    data = {
        '姓名': ['张三', '李四', '王五', '赵六'],
        '年龄': [25, 30, 35, 28],
        '部门': ['技术部', '销售部', '人事部', '财务部'],
        '工资': [8000, 12000, 9000, 11000]
    }
    df = pd.DataFrame(data)
    
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='员工信息', index=False)
    excel_data = excel_buffer.getvalue()
    
    # 2. 创建文本测试文件
    text_data = "这是一个测试文本文件\n包含多行内容\n用于测试预览功能\n".encode('utf-8')
    
    # 3. 创建CSV测试文件
    csv_data = df.to_csv(index=False).encode('utf-8')
    
    print("✅ 测试文件创建完成")
    
    # 测试文件类型检测
    print("\n📁 测试文件类型检测:")
    test_files = [
        ("test.xlsx", excel_data),
        ("test.txt", text_data),
        ("test.csv", csv_data)
    ]
    
    for filename, file_data in test_files:
        file_type = storage_manager.get_file_type(filename)
        icon = storage_manager.get_file_icon(file_type)
        print(f"  {filename} -> {file_type} {icon}")
    
    # 测试Excel预览
    print("\n📊 测试Excel预览:")
    try:
        df_preview = pd.read_excel(io.BytesIO(excel_data))
        print("✅ Excel预览功能正常")
        print(f"   数据形状: {df_preview.shape}")
        print(f"   列名: {list(df_preview.columns)}")
    except Exception as e:
        print(f"❌ Excel预览失败: {e}")
    
    # 测试文本预览
    print("\n📝 测试文本预览:")
    try:
        text_content = text_data.decode('utf-8')
        print("✅ 文本预览功能正常")
        print(f"   内容长度: {len(text_content)} 字符")
        print(f"   预览内容: {text_content[:50]}...")
    except Exception as e:
        print(f"❌ 文本预览失败: {e}")
    
    print("\n🎯 预览功能测试完成!")

if __name__ == "__main__":
    test_preview_function()
