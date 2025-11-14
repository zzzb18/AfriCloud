#!/usr/bin/env python3
"""
AI Cloud Storage 启动脚本
避免fitz模块导入问题
"""

import os
import sys

# 设置环境变量，避免某些模块的警告
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# 导入并运行应用
if __name__ == "__main__":
    try:
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
        sys.exit(stcli.main())
    except Exception as e:
        print(f"启动失败: {e}")
        print("请尝试运行: python -m streamlit run app.py")
