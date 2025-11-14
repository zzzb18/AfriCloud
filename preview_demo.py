#!/usr/bin/env python3
"""
预览功能演示页面
"""

import streamlit as st
import pandas as pd
import io
from app import CloudStorageManager

st.set_page_config(
    page_title="预览功能演示",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ 预览功能演示")

# 初始化存储管理器
if 'storage_manager' not in st.session_state:
    st.session_state.storage_manager = CloudStorageManager()

storage_manager = st.session_state.storage_manager

# 创建测试文件
st.markdown("### 📄 创建测试文件")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 创建Excel测试文件"):
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
        
        # 模拟上传文件
        class MockFile:
            def __init__(self, name, data):
                self.name = name
                self.data = data
            
            def getbuffer(self):
                return self.data
        
        mock_file = MockFile("test_employees.xlsx", excel_data)
        result = storage_manager.upload_file(mock_file)
        
        if result["success"]:
            st.success("✅ Excel测试文件创建成功!")
        else:
            st.error(f"❌ 创建失败: {result['error']}")

with col2:
    if st.button("📝 创建文本测试文件"):
        text_content = """这是一个测试文本文件
包含多行内容
用于测试预览功能

功能特点:
- 支持多种文件格式
- 实时预览
- 无需下载
- 智能识别文件类型"""
        
        text_data = text_content.encode('utf-8')
        
        class MockFile:
            def __init__(self, name, data):
                self.name = name
                self.data = data
            
            def getbuffer(self):
                return self.data
        
        mock_file = MockFile("test_document.txt", text_data)
        result = storage_manager.upload_file(mock_file)
        
        if result["success"]:
            st.success("✅ 文本测试文件创建成功!")
        else:
            st.error(f"❌ 创建失败: {result['error']}")

with col3:
    if st.button("📊 创建CSV测试文件"):
        data = {
            '产品': ['iPhone', 'Samsung', 'Huawei', 'Xiaomi'],
            '价格': [6999, 5999, 4999, 2999],
            '销量': [100, 150, 120, 200],
            '评分': [4.8, 4.6, 4.7, 4.5]
        }
        df = pd.DataFrame(data)
        csv_data = df.to_csv(index=False).encode('utf-8')
        
        class MockFile:
            def __init__(self, name, data):
                self.name = name
                self.data = data
            
            def getbuffer(self):
                return self.data
        
        mock_file = MockFile("test_products.csv", csv_data)
        result = storage_manager.upload_file(mock_file)
        
        if result["success"]:
            st.success("✅ CSV测试文件创建成功!")
        else:
            st.error(f"❌ 创建失败: {result['error']}")

# 显示文件列表
st.markdown("### 📁 文件列表")

files = storage_manager.get_files()

if files:
    for file in files:
        with st.expander(f"{storage_manager.get_file_icon(file['file_type'])} {file['filename']} ({storage_manager.format_file_size(file['file_size'])})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**文件类型**: {file['file_type']}")
                st.write(f"**上传时间**: {file['upload_time']}")
                st.write(f"**缓存状态**: {'✅ 已缓存' if file['is_cached'] else '☁️ 云端'}")
            
            with col2:
                # 使用checkbox来控制预览状态
                show_preview = st.checkbox("👁️ 预览", key=f"preview_demo_{file['id']}", help="点击预览文件内容")
            
            # 显示预览内容
            if show_preview:
                st.markdown("#### 📄 文件预览")
                file_data = storage_manager.preview_file(file['id'])
                
                if file_data:
                    if file['file_type'] == 'image':
                        st.image(file_data, caption=file['filename'])
                    elif file['file_type'] == 'application' and file['filename'].endswith('.pdf'):
                        st.info("PDF预览功能需要PyMuPDF模块")
                        st.download_button(
                            "📥 下载PDF",
                            file_data,
                            file['filename'],
                            key=f"demo_download_pdf_{file['id']}"
                        )
                    elif file['file_type'] == 'application' and file['filename'].endswith(('.xlsx', '.xls')):
                        try:
                            df = pd.read_excel(io.BytesIO(file_data))
                            st.dataframe(df, use_container_width=True)
                            st.caption(f"Excel预览: {file['filename']}")
                        except Exception as e:
                            st.error(f"Excel预览失败: {str(e)}")
                    elif file['file_type'] == 'text' or file['filename'].endswith('.txt'):
                        try:
                            text_content = file_data.decode('utf-8')
                            st.text_area("文件内容", text_content, height=200)
                            st.caption(f"文本预览: {file['filename']}")
                        except Exception as e:
                            st.error(f"文本预览失败: {str(e)}")
                    elif file['filename'].endswith('.csv'):
                        try:
                            df = pd.read_csv(io.BytesIO(file_data))
                            st.dataframe(df, use_container_width=True)
                            st.caption(f"CSV预览: {file['filename']}")
                        except Exception as e:
                            st.error(f"CSV预览失败: {str(e)}")
                    else:
                        st.info(f"暂不支持 {file['file_type']} 类型的预览")
                        st.download_button(
                            "📥 下载文件",
                            file_data,
                            file['filename'],
                            key=f"demo_download_{file['id']}"
                        )
                else:
                    st.error("无法预览此文件")
else:
    st.info("📁 暂无文件，请先创建测试文件")

# 使用说明
st.markdown("### 📋 使用说明")
st.info("""
**预览功能说明:**
1. 点击"创建测试文件"按钮创建不同类型的测试文件
2. 在文件列表中点击"👁️ 预览"按钮查看文件内容
3. 支持的文件类型:
   - 📊 Excel文件 (.xlsx, .xls) - 显示数据表格
   - 📝 文本文件 (.txt) - 显示文本内容
   - 📊 CSV文件 (.csv) - 显示数据表格
   - 🖼️ 图片文件 - 显示图片
   - 📄 PDF文件 - 需要PyMuPDF模块
4. 点击"❌ 关闭预览"按钮关闭预览内容
""")
