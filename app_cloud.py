#!/usr/bin/env python3
"""
云存储系统 - 云部署版本
支持远程访问和多人使用
"""

import streamlit as st
import sqlite3
import os
import shutil
import hashlib
import mimetypes
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import io

# 页面配置
st.set_page_config(
    page_title="☁️ AI Cloud Storage",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 云部署检测
IS_CLOUD_DEPLOYMENT = os.getenv('STREAMLIT_SERVER_PORT') is not None

# 存储路径配置
if IS_CLOUD_DEPLOYMENT:
    STORAGE_BASE = Path("/tmp/cloud_storage")
    CACHE_BASE = Path("/tmp/local_cache")
    AI_BASE = Path("/tmp/ai_analysis")
else:
    STORAGE_BASE = Path("cloud_storage")
    CACHE_BASE = Path("local_cache")
    AI_BASE = Path("ai_analysis")

# 创建目录
STORAGE_BASE.mkdir(exist_ok=True)
CACHE_BASE.mkdir(exist_ok=True)
AI_BASE.mkdir(exist_ok=True)

class CloudStorageManager:
    def __init__(self):
        self.storage_dir = STORAGE_BASE
        self.cache_dir = CACHE_BASE
        self.ai_analysis_dir = AI_BASE
        self.db_path = self.storage_dir / "storage.db"
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 文件表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                file_type TEXT,
                folder_id INTEGER,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum TEXT,
                is_cached BOOLEAN DEFAULT FALSE,
                user_id TEXT DEFAULT 'default',
                FOREIGN KEY (folder_id) REFERENCES folders (id)
            )
        ''')
        
        # 文件夹表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                folder_name TEXT NOT NULL,
                parent_folder_id INTEGER,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT DEFAULT 'default',
                FOREIGN KEY (parent_folder_id) REFERENCES folders (id)
            )
        ''')
        
        # 用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def upload_file(self, uploaded_file, folder_id: Optional[int] = None, user_id: str = 'default') -> Dict[str, Any]:
        """上传文件"""
        try:
            # 生成唯一文件名
            timestamp = int(time.time())
            filename = f"{timestamp}_{uploaded_file.name}"
            file_path = self.storage_dir / filename
            
            # 保存文件
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 计算文件信息
            file_size = file_path.stat().st_size
            checksum = self.calculate_checksum(str(file_path))
            file_type = self.get_file_type(uploaded_file.name)
            
            # 保存到数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO files (filename, file_path, file_size, file_type, folder_id, checksum, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (uploaded_file.name, str(file_path), file_size, file_type, folder_id, checksum, user_id))
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "filename": uploaded_file.name,
                "file_size": file_size,
                "file_type": file_type,
                "cloud_url": f"https://your-domain.com/files/{filename}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_files(self, folder_id: Optional[int] = None, user_id: str = 'default') -> List[Dict[str, Any]]:
        """获取文件列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if folder_id is None:
            cursor.execute('''
                SELECT id, filename, file_size, file_type, upload_time, is_cached
                FROM files WHERE folder_id IS NULL AND user_id = ?
                ORDER BY upload_time DESC
            ''', (user_id,))
        else:
            cursor.execute('''
                SELECT id, filename, file_size, file_type, upload_time, is_cached
                FROM files WHERE folder_id = ? AND user_id = ?
                ORDER BY upload_time DESC
            ''', (folder_id, user_id))
        
        files = []
        for row in cursor.fetchall():
            files.append({
                "id": row[0],
                "filename": row[1],
                "file_size": row[2],
                "file_type": row[3],
                "upload_time": row[4],
                "is_cached": bool(row[5])
            })
        
        conn.close()
        return files
    
    def calculate_checksum(self, file_path: str) -> str:
        """计算文件校验和"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_file_type(self, filename: str) -> str:
        """获取文件类型"""
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type:
            return mime_type.split('/')[0]
        return 'unknown'
    
    def format_file_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f}{size_names[i]}"
    
    def get_file_icon(self, file_type: str) -> str:
        """获取文件类型图标"""
        icons = {
            'image': '🖼️',
            'application': '📄',
            'text': '📝',
            'video': '🎥',
            'audio': '🎵',
            'unknown': '📁'
        }
        return icons.get(file_type, '📁')

# 初始化存储管理器
if 'storage_manager' not in st.session_state:
    st.session_state.storage_manager = CloudStorageManager()

storage_manager = st.session_state.storage_manager

# 用户认证（简单版本）
if 'user_id' not in st.session_state:
    st.session_state.user_id = 'default'

# 主界面
st.title("☁️ AI Cloud Storage")
st.markdown("**Cloud-based file management with AI capabilities**")

# 云部署状态
if IS_CLOUD_DEPLOYMENT:
    st.success("🌐 **Cloud Deployment Active** - Accessible from anywhere!")
else:
    st.info("💻 **Local Deployment** - Running on localhost")

# 侧边栏
with st.sidebar:
    st.markdown("### 📁 File Management")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'xlsx', 'xls', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'csv', 'txt', 'doc', 'docx']
    )
    
    if uploaded_file:
        if st.button("📤 Upload to Cloud"):
            with st.spinner("Uploading to cloud..."):
                result = storage_manager.upload_file(uploaded_file, user_id=st.session_state.user_id)
                if result["success"]:
                    st.success(f"✅ {result['filename']} uploaded successfully!")
                    st.info(f"☁️ Cloud URL: {result['cloud_url']}")
                else:
                    st.error(f"❌ Upload failed: {result['error']}")
    
    st.markdown("---")
    
    # 文件统计
    files = storage_manager.get_files(user_id=st.session_state.user_id)
    total_size = sum(file.get('file_size', 0) for file in files)
    cached_count = sum(1 for file in files if file.get('is_cached', False))
    
    st.metric("Total Files", len(files))
    st.metric("Total Size", storage_manager.format_file_size(total_size))
    st.metric("Cached Files", f"{cached_count}/{len(files)}")

# 主内容区域
if files:
    st.markdown("### 📄 Your Files")
    
    # 文件列表
    for file in files:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"{storage_manager.get_file_icon(file['file_type'])} **{file['filename']}**")
                st.caption(f"Type: {file['file_type']} | Uploaded: {file['upload_time']}")
            
            with col2:
                st.write(f"📏 {storage_manager.format_file_size(file['file_size'])}")
            
            with col3:
                status = "✅ Cached" if file['is_cached'] else "☁️ Cloud"
                st.write(f"**{status}**")
            
            st.markdown("---")
else:
    st.info("No files uploaded yet. Upload a file using the sidebar!")

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b;'>
    <p>☁️ AI Cloud Storage - Deployed on Cloud</p>
    <p>Access your files from anywhere in the world!</p>
</div>
""", unsafe_allow_html=True)

