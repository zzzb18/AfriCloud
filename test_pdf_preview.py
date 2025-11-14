#!/usr/bin/env python3
"""
测试PDF预览功能
"""

try:
    import fitz
    print("✅ PyMuPDF (fitz) 模块已安装")
    
    # 测试PDF预览功能
    print("📄 测试PDF预览功能...")
    
    # 创建一个简单的PDF文件用于测试
    doc = fitz.open()  # 创建新文档
    page = doc.new_page()  # 添加页面
    page.insert_text((100, 100), "Hello, AI Cloud Storage!")  # 添加文本
    page.insert_text((100, 150), "PDF预览功能测试")
    
    # 保存测试PDF
    test_pdf_path = "test_document.pdf"
    doc.save(test_pdf_path)
    doc.close()
    
    print(f"✅ 测试PDF已创建: {test_pdf_path}")
    
    # 测试读取PDF
    doc = fitz.open(test_pdf_path)
    page = doc[0]
    pix = page.get_pixmap()
    img_data = pix.tobytes("png")
    
    print(f"✅ PDF预览功能正常，图片大小: {len(img_data)} 字节")
    
    doc.close()
    
except ImportError:
    print("❌ PyMuPDF (fitz) 模块未安装")
    print("请运行: pip install PyMuPDF")
except Exception as e:
    print(f"❌ PDF预览测试失败: {e}")

print("\n🎯 PDF预览功能测试完成!")
