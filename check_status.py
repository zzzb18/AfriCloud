#!/usr/bin/env python3
"""
检查AI Cloud Storage应用状态
"""

import requests
import time

def check_app_status():
    """检查应用状态"""
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ AI Cloud Storage 应用正在运行")
            print("🌐 访问地址: http://localhost:8501")
            return True
        else:
            print(f"❌ 应用响应异常: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到应用，请确保应用正在运行")
        return False
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

if __name__ == "__main__":
    print("🔍 检查 AI Cloud Storage 应用状态...")
    check_app_status()
