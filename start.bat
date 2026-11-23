@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   KINETIC-X // 机械工业人体骨骼与手部识别系统 (Server Ready)
echo   自动识别本机硬件: CUDA GPU / 多核 CPU 自适应
echo   监听地址: 0.0.0.0:8000 (支持局域网与外网设备访问)
echo ============================================================
".venv\Scripts\python.exe" server.py --host 0.0.0.0 --port 8000 --device auto
pause
