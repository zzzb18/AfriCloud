@echo off
echo 启动 AI Cloud Storage...
echo.
echo 正在启动应用，请稍候...
echo 应用将在浏览器中自动打开: http://localhost:8501
echo.
echo 按 Ctrl+C 可以停止应用
echo.

python -m streamlit run app.py --server.port=8501 --server.address=0.0.0.0

pause
