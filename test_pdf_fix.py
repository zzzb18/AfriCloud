#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF预览功能测试和修复
"""

import streamlit as st
import io
import sys

def test_pdf_preview():
    """测试PDF预览功能"""
    st.title("🔍 PDF预览功能测试")
    
    # 检查PyMuPDF是否可用
    try:
        import fitz
        st.success("✅ PyMuPDF (fitz) 模块已安装")
        PDF_AVAILABLE = True
    except ImportError:
        st.error("❌ PyMuPDF (fitz) 模块未安装")
        st.info("请运行: pip install PyMuPDF")
        PDF_AVAILABLE = False
        return
    
    # 测试PDF文件上传
    uploaded_file = st.file_uploader(
        "上传PDF文件进行测试",
        type=['pdf'],
        help="选择一个PDF文件来测试预览功能"
    )
    
    if uploaded_file is not None:
        st.info(f"📄 已选择文件: {uploaded_file.name}")
        st.info(f"📊 文件大小: {len(uploaded_file.getbuffer())} 字节")
        
        # 读取文件数据
        file_data = uploaded_file.getbuffer()
        
        # 测试PDF预览
        st.markdown("### 🔍 PDF预览测试")
        
        try:
            # 使用BytesIO包装数据
            pdf_stream = io.BytesIO(file_data)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            
            st.success(f"✅ PDF文件成功打开，共 {len(doc)} 页")
            
            # 获取第一页
            page = doc[0]
            st.info(f"📄 第一页尺寸: {page.rect.width} x {page.rect.height}")
            
            # 渲染为图片
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            
            st.success("✅ PDF页面成功渲染为图片")
            st.info(f"🖼️ 图片数据大小: {len(img_data)} 字节")
            
            # 显示图片
            st.image(img_data, caption=f"PDF预览: {uploaded_file.name} (第1页)", use_column_width=True)
            
            # 关闭文档
            doc.close()
            st.success("✅ PDF文档已正确关闭")
            
        except Exception as e:
            st.error(f"❌ PDF预览失败: {str(e)}")
            st.error(f"❌ 错误类型: {type(e).__name__}")
            
            # 显示详细错误信息
            import traceback
            st.code(traceback.format_exc())
            
            # 提供下载选项
            st.download_button(
                "📥 下载PDF文件",
                file_data,
                uploaded_file.name,
                key="download_pdf_test"
            )

if __name__ == "__main__":
    test_pdf_preview()
