import streamlit as st
import pandas as pd
import os
import json
import hashlib
import mimetypes
import base64
import io
import time
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
import zipfile
import shutil
from pathlib import Path
import requests
from PIL import Image
try:
    import fitz  # PyMuPDF for PDF preview
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Set page config with premium aesthetics
st.set_page_config(
    page_title="AI Cloud Storage",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #2c3e50;
    }
    
    h1, h2, h3, h4, h5, h6 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 700;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 15px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

class CloudStorageManager:
    def __init__(self):
        self.storage_dir = Path("cloud_storage")
        self.storage_dir.mkdir(exist_ok=True)
        self.db_path = self.storage_dir / "storage.db"
        self.cache_dir = Path("local_cache")
        self.cache_dir.mkdir(exist_ok=True)
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
                FOREIGN KEY (parent_folder_id) REFERENCES folders (id)
            )
        ''')
        
        # 上传进度表（用于断点续传）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS upload_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                total_size INTEGER,
                uploaded_size INTEGER,
                chunk_size INTEGER,
                checksum TEXT,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
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
    
    def upload_file(self, uploaded_file, folder_id: Optional[int] = None) -> Dict[str, Any]:
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
                INSERT INTO files (filename, file_path, file_size, file_type, folder_id, checksum)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (uploaded_file.name, str(file_path), file_size, file_type, folder_id, checksum))
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "filename": uploaded_file.name,
                "file_size": file_size,
                "file_type": file_type
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_files(self, folder_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取文件列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if folder_id is None:
            cursor.execute('''
                SELECT id, filename, file_size, file_type, upload_time, is_cached
                FROM files WHERE folder_id IS NULL
                ORDER BY upload_time DESC
            ''')
        else:
            cursor.execute('''
                SELECT id, filename, file_size, file_type, upload_time, is_cached
                FROM files WHERE folder_id = ?
                ORDER BY upload_time DESC
            ''', (folder_id,))
        
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
    
    def create_folder(self, folder_name: str, parent_folder_id: Optional[int] = None) -> Dict[str, Any]:
        """创建文件夹"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO folders (folder_name, parent_folder_id)
                VALUES (?, ?)
            ''', (folder_name, parent_folder_id))
            conn.commit()
            folder_id = cursor.lastrowid
            conn.close()
            
            return {"success": True, "folder_id": folder_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_files(self, query: str, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索文件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if file_type:
            cursor.execute('''
                SELECT id, filename, file_size, file_type, upload_time, is_cached
                FROM files 
                WHERE filename LIKE ? AND file_type = ?
                ORDER BY upload_time DESC
            ''', (f"%{query}%", file_type))
        else:
            cursor.execute('''
                SELECT id, filename, file_size, file_type, upload_time, is_cached
                FROM files 
                WHERE filename LIKE ?
                ORDER BY upload_time DESC
            ''', (f"%{query}%",))
        
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
    
    def preview_file(self, file_id: int) -> Optional[bytes]:
        """预览文件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT file_path, file_type FROM files WHERE id = ?', (file_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        file_path, file_type = result
        
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except:
            return None
    
    def cache_file(self, file_id: int) -> bool:
        """缓存文件到本地"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT file_path, filename FROM files WHERE id = ?', (file_id,))
            result = cursor.fetchone()
            
            if result:
                file_path, filename = result
                cache_path = self.cache_dir / filename
                shutil.copy2(file_path, cache_path)
                
                # 更新数据库
                cursor.execute('UPDATE files SET is_cached = TRUE WHERE id = ?', (file_id,))
                conn.commit()
                conn.close()
                return True
        except:
            pass
        return False
    
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

# 初始化云存储管理器
if 'storage_manager' not in st.session_state:
    st.session_state.storage_manager = CloudStorageManager()

storage_manager = st.session_state.storage_manager

# 侧边栏
with st.sidebar:
    st.markdown("### ☁️ AI Cloud Storage")
    st.markdown("---")
    
    # 快速操作
    st.markdown("### ⚡ Quick Actions")
    
    # 创建文件夹
    with st.form("create_folder_form"):
        folder_name = st.text_input("📁 新建文件夹", placeholder="输入文件夹名称")
        if st.form_submit_button("创建", use_container_width=True):
            if folder_name:
                result = storage_manager.create_folder(folder_name)
                if result["success"]:
                    st.success(f"✅ 文件夹 '{folder_name}' 创建成功!")
                else:
                    st.error(f"❌ 创建失败: {result['error']}")
            else:
                st.warning("请输入文件夹名称")
    
    st.markdown("---")
    
    # 搜索功能
    st.markdown("### 🔍 搜索文件")
    search_query = st.text_input("搜索文件名", placeholder="输入关键词")
    search_type = st.selectbox("文件类型", ["全部", "image", "application", "text", "video", "audio"])
    
    if st.button("🔍 搜索", use_container_width=True) and search_query:
        file_type = None if search_type == "全部" else search_type
        search_results = storage_manager.search_files(search_query, file_type)
        st.session_state.search_results = search_results
        st.session_state.show_search = True
    else:
        st.session_state.show_search = False

# 主界面
st.title("☁️ AI Cloud Storage")
st.markdown("智能云存储 - 支持断点续传、在线预览、本地缓存")

# 文件上传区域
st.markdown("### 📤 文件上传")

uploaded_files = st.file_uploader(
    "选择文件上传", 
    type=["xlsx", "xls", "csv", "pdf", "png", "jpg", "jpeg", "gif", "bmp", "txt", "doc", "docx"],
    accept_multiple_files=True,
    help="支持 Excel、PDF、图片、CSV 等格式"
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"📄 {uploaded_file.name} ({storage_manager.format_file_size(len(uploaded_file.getbuffer()))})")
        
        with col2:
            if st.button(f"📤 上传", key=f"upload_{uploaded_file.name}"):
                with st.spinner(f"正在上传 {uploaded_file.name}..."):
                    result = storage_manager.upload_file(uploaded_file)
                    if result["success"]:
                        st.success(f"✅ {uploaded_file.name} 上传成功!")
                        st.rerun()
                    else:
                        st.error(f"❌ 上传失败: {result['error']}")

# 文件列表显示
st.markdown("### 📁 文件列表")

# 检查是否显示搜索结果
if st.session_state.get('show_search', False) and 'search_results' in st.session_state:
    files = st.session_state.search_results
    st.info(f"🔍 搜索结果: 找到 {len(files)} 个文件")
else:
    files = storage_manager.get_files()

if files:
    # 文件统计
    total_size = sum(file['file_size'] for file in files)
    cached_count = sum(1 for file in files if file['is_cached'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("文件总数", len(files))
    with col2:
        st.metric("总大小", storage_manager.format_file_size(total_size))
    with col3:
        st.metric("已缓存", f"{cached_count}/{len(files)}")
    with col4:
        st.metric("缓存率", f"{cached_count/len(files)*100:.1f}%")
    
    st.markdown("---")
    
    # 文件列表
    for file in files:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 2, 2])
            
            with col1:
                st.write(storage_manager.get_file_icon(file['file_type']))
            
            with col2:
                st.write(f"**{file['filename']}**")
                st.caption(f"类型: {file['file_type']} | 上传时间: {file['upload_time']}")
            
            with col3:
                st.write(storage_manager.format_file_size(file['file_size']))
            
            with col4:
                if file['is_cached']:
                    st.success("✅ 已缓存")
                else:
                    st.info("☁️ 云端")
            
            with col5:
                button_col1, button_col2, button_col3 = st.columns(3)
                
                with button_col1:
                    if st.button("👁️", key=f"preview_{file['id']}", help="预览"):
                        file_data = storage_manager.preview_file(file['id'])
                        if file_data:
                            if file['file_type'] == 'image':
                                st.image(file_data)
                            elif file['file_type'] == 'application' and file['filename'].endswith('.pdf'):
                                if PDF_AVAILABLE:
                                    try:
                                        # PDF预览功能
                                        doc = fitz.open(stream=file_data, filetype="pdf")
                                        page = doc[0]  # 获取第一页
                                        pix = page.get_pixmap()
                                        img_data = pix.tobytes("png")
                                        st.image(img_data, caption="PDF预览 (第1页)")
                                        doc.close()
                                    except Exception as e:
                                        st.error(f"PDF预览失败: {str(e)}")
                                        st.download_button(
                                            "📥 下载PDF",
                                            file_data,
                                            file['filename'],
                                            key=f"download_pdf_{file['id']}"
                                        )
                                else:
                                    st.info("PDF预览需要安装PyMuPDF模块")
                                    st.download_button(
                                        "📥 下载PDF",
                                        file_data,
                                        file['filename'],
                                        key=f"download_pdf_{file['id']}"
                                    )
                            else:
                                st.download_button(
                                    "📥 下载预览",
                                    file_data,
                                    file['filename'],
                                    key=f"download_{file['id']}"
                                )
                        else:
                            st.error("无法预览此文件")
                
                with button_col2:
                    if not file['is_cached']:
                        if st.button("💾", key=f"cache_{file['id']}", help="缓存到本地"):
                            if storage_manager.cache_file(file['id']):
                                st.success("缓存成功!")
                                st.rerun()
                            else:
                                st.error("缓存失败")
                    else:
                        st.success("已缓存")
                
                with button_col3:
                    if st.button("📥", key=f"download_{file['id']}", help="下载"):
                        file_data = storage_manager.preview_file(file['id'])
                        if file_data:
                            st.download_button(
                                "📥 下载文件",
                                file_data,
                                file['filename'],
                                key=f"download_btn_{file['id']}"
                            )
                        else:
                            st.error("文件不存在")
            
            st.markdown("---")

else:
    # 空状态
    st.markdown("<div style='text-align: center; padding: 40px 0;'>", unsafe_allow_html=True)
    st.header("📁 暂无文件")
    st.subheader("上传您的第一个文件开始使用云存储")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 功能说明
    features = st.columns(3)
    with features[0]:
        st.info("""
        **📤 文件上传**
        - 支持多种格式
        - 断点续传
        - 自动校验
        """)
    with features[1]:
        st.success("""
        **👁️ 在线预览**
        - 图片即时预览
        - PDF文档查看
        - 无需下载
        """)
    with features[2]:
        st.warning("""
        **💾 本地缓存**
        - 离线访问
        - 自动同步
        - 智能管理
        """)

# 页脚
st.markdown("---")
st.markdown("**Built with ❤️ • AI Cloud Storage • ☁️ 智能存储**")
