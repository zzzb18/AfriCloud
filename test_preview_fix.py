#!/usr/bin/env python3
"""
测试预览功能修复
"""

import streamlit as st
import pandas as pd
import io
from app import CloudStorageManager

# 设置页面配置
st.set_page_config(
    page_title="预览功能测试",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 预览功能测试")

# 初始化存储管理器
if 'storage_manager' not in st.session_state:
    st.session_state.storage_manager = CloudStorageManager()

storage_manager = st.session_state.storage_manager

# 创建测试文件
if st.button("📄 创建测试文件"):
    # 创建Excel文件
    data = {
        '姓名': ['张三', '李四', '王五'],
        '年龄': [25, 30, 35],
        '部门': ['技术部', '销售部', '人事部']
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
    
    mock_file = MockFile("test.xlsx", excel_data)
    result = storage_manager.upload_file(mock_file)
    
    if result["success"]:
        st.success("✅ 测试文件创建成功!")
    else:
        st.error(f"❌ 创建失败: {result['error']}")

# 显示文件列表
files = storage_manager.get_files()

if files:
    st.markdown("### 📁 文件列表")
    
    for file in files:
        with st.expander(f"📄 {file['filename']} ({storage_manager.format_file_size(file['file_size'])})"):
            # 使用checkbox控制预览
            show_preview = st.checkbox("👁️ 预览文件", key=f"preview_{file['id']}")
            
            if show_preview:
                st.markdown("#### 📄 文件预览")
                file_data = storage_manager.preview_file(file['id'])
                
                if file_data:
                    if file['file_type'] == 'application' and file['filename'].endswith(('.xlsx', '.xls')):
                        try:
                            df = pd.read_excel(io.BytesIO(file_data))
                            st.dataframe(df, use_container_width=True)
                            st.caption(f"Excel预览: {file['filename']}")
                        except Exception as e:
                            st.error(f"Excel预览失败: {str(e)}")
                    else:
                        st.info(f"文件类型: {file['file_type']}")
                        st.download_button(
                            "📥 下载文件",
                            file_data,
                            file['filename'],
                            key=f"download_{file['id']}"
                        )
                else:
                    st.error("无法预览此文件")
else:
    st.info("📁 暂无文件，请先创建测试文件")

st.markdown("### 📋 使用说明")
st.info("""
**预览功能测试说明:**
1. 点击"📄 创建测试文件"按钮创建测试文件
2. 在文件列表中勾选"👁️ 预览文件"复选框查看文件内容
3. 支持Excel文件的数据表格预览
4. 其他文件类型提供下载功能
""")
