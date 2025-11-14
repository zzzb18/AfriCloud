#!/usr/bin/env python3
"""
测试Excel文件预览功能
"""

import pandas as pd
import io
import streamlit as st

def test_excel_preview():
    """测试Excel文件预览"""
    print("📊 测试Excel文件预览功能...")
    
    # 创建测试数据
    data = {
        '姓名': ['张三', '李四', '王五', '赵六'],
        '年龄': [25, 30, 35, 28],
        '部门': ['技术部', '销售部', '人事部', '财务部'],
        '工资': [8000, 12000, 9000, 11000]
    }
    
    df = pd.DataFrame(data)
    
    # 创建Excel文件
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='员工信息', index=False)
    
    excel_data = excel_buffer.getvalue()
    
    print(f"✅ 测试Excel文件已创建，大小: {len(excel_data)} 字节")
    print("📋 数据预览:")
    print(df.head())
    
    # 测试文件类型检测
    from app import CloudStorageManager
    storage_manager = CloudStorageManager()
    
    file_type = storage_manager.get_file_type("test.xlsx")
    print(f"📁 文件类型检测: {file_type}")
    
    icon = storage_manager.get_file_icon(file_type)
    print(f"🎯 文件图标: {icon}")
    
    print("✅ Excel预览功能测试完成!")

if __name__ == "__main__":
    test_excel_preview()
