#!/bin/bash

# 云存储部署脚本
echo "🚀 开始部署云存储系统..."

# 检查是否安装了 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

# 检查是否安装了 Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose 未安装，请先安装 Docker Compose"
    exit 1
fi

# 创建必要的目录
echo "📁 创建存储目录..."
mkdir -p cloud_data cache_data ai_data

# 设置权限
echo "🔐 设置目录权限..."
chmod 755 cloud_data cache_data ai_data

# 构建和启动服务
echo "🔨 构建 Docker 镜像..."
docker-compose build

echo "🚀 启动服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo "📊 检查服务状态..."
docker-compose ps

# 显示访问信息
echo ""
echo "✅ 部署完成！"
echo "🌐 访问地址: http://localhost:8501"
echo "📊 查看日志: docker-compose logs -f"
echo "🛑 停止服务: docker-compose down"
echo ""

# 显示服务状态
if docker-compose ps | grep -q "Up"; then
    echo "🎉 服务运行正常！"
else
    echo "❌ 服务启动失败，请检查日志"
    docker-compose logs
fi

