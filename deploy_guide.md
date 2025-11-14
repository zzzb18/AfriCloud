# ☁️ 云存储部署指南

## 🚀 部署方案

### 方案1：Streamlit Cloud（免费，推荐新手）

**步骤**：
1. 将代码推送到 GitHub 仓库
2. 访问 [share.streamlit.io](https://share.streamlit.io)
3. 点击 "New app"
4. 选择 GitHub 仓库和 `app.py` 文件
5. 点击 "Deploy"

**优点**：
- 完全免费
- 自动部署和更新
- 无需服务器管理

**限制**：
- 文件存储临时性（重启丢失）
- 有使用时间限制

### 方案2：云服务器部署（推荐生产环境）

#### 2.1 使用 Docker（推荐）

**在云服务器上**：
```bash
# 克隆代码
git clone <your-repo-url>
cd <project-directory>

# 使用 Docker Compose 部署
docker-compose up -d

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

**访问**：`http://your-server-ip:8501`

#### 2.2 直接部署

**在云服务器上**：
```bash
# 安装 Python 3.9+
sudo apt update
sudo apt install python3.9 python3.9-pip

# 安装依赖
pip3 install -r requirements.txt

# 运行应用
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

### 方案3：云平台部署

#### 3.1 Heroku
```bash
# 安装 Heroku CLI
# 创建 Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# 部署
heroku create your-app-name
git push heroku main
```

#### 3.2 Railway
1. 连接 GitHub 仓库
2. 选择 `app.py` 作为入口文件
3. 自动部署

#### 3.3 DigitalOcean App Platform
1. 连接 GitHub 仓库
2. 选择 Python 应用
3. 设置启动命令：`streamlit run app.py --server.port=8080 --server.address=0.0.0.0`

## 🔧 配置说明

### 环境变量
```bash
# 云部署标识
STREAMLIT_SERVER_PORT=8501

# 数据库路径（云部署时使用 /tmp）
CLOUD_STORAGE_PATH=/tmp/cloud_storage

# 缓存路径
CACHE_PATH=/tmp/local_cache
```

### 持久化存储
- **本地部署**：数据保存在项目目录
- **云部署**：数据保存在 `/tmp` 目录（需要配置持久化卷）

## 📊 性能优化

### 1. 数据库优化
```python
# 在 init_database 中添加索引
cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_folder ON files(folder_id)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_type ON files(file_type)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_cached ON files(is_cached)')
```

### 2. 文件存储优化
- 使用云存储服务（AWS S3、阿里云OSS等）
- 实现文件分片存储
- 添加CDN加速

### 3. 缓存优化
- 使用Redis缓存
- 实现分布式缓存
- 添加缓存过期策略

## 🔒 安全配置

### 1. 访问控制
```python
# 在 app.py 开头添加
import streamlit as st

# 简单的密码保护
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("Enter password", type="password")
    if st.button("Login"):
        if password == "your_password":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Wrong password")
    st.stop()
```

### 2. HTTPS 配置
```bash
# 使用 nginx 反向代理
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📈 监控和维护

### 1. 日志监控
```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### 2. 健康检查
```python
# 添加健康检查端点
@app.route('/health')
def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}
```

### 3. 备份策略
- 定期备份数据库
- 备份重要文件
- 实现增量备份

## 🎯 推荐部署架构

```
用户 → CDN → 负载均衡器 → 应用服务器 → 数据库
                    ↓
                文件存储服务
```

**组件**：
- **前端**：Streamlit 应用
- **数据库**：PostgreSQL（生产环境）
- **文件存储**：云存储服务
- **缓存**：Redis
- **监控**：Prometheus + Grafana

## 💰 成本估算

### Streamlit Cloud
- **免费**：适合测试和小规模使用

### 云服务器（月费用）
- **阿里云/腾讯云**：50-200元/月
- **AWS EC2**：$10-50/月
- **DigitalOcean**：$5-20/月

### 云存储服务
- **阿里云OSS**：0.12元/GB/月
- **AWS S3**：$0.023/GB/月

## 🚀 快速开始

1. **选择部署方案**
2. **准备云服务器**（如选择方案2）
3. **上传代码**
4. **安装依赖**
5. **启动服务**
6. **配置域名和HTTPS**（可选）

现在您的云存储系统就可以在互联网上访问了！🌐

