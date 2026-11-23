# KINETIC-X // 机械工业风人体骨骼 & 手部识别系统 (Server Ready v2.0)

支持**本地 GPU 工作站**与**云端/内网服务器部署**的网页交互式人体骨骼与 21 点手部识别控制台。启动时自动识别宿主机硬件环境（NVIDIA CUDA / Apple MPS / 多核 CPU）并匹配最优推理参数，同时针对外部网络（局域网 / 公网 / 手机平板）进行低延迟流畅传输优化。

## 核心特性

- **硬件环境自动识别与自适应配置**：
  - 自动探测 `NVIDIA CUDA`（区分 8GB+ 高显存与 6GB 标准显存档位）、`Apple MPS` 与 `CPU` 核心数/内存，自动选择计算设备、推荐张量分辨率（`320 / 480 / 640px`）与推荐推流规格。
  - 支持环境变量或命令行手动覆盖：`--device auto|cuda|cpu`、`KINETIC_DEVICE=auto`。
- **人体骨骼 + 21 点手部双引擎独立开关**：
  - **人体骨骼识别**：YOLO11-Nano / YOLO11-Small Pose 多人 17 点检测 + 9 项关节角度与姿态分类。
  - **手部骨骼识别**：MediaPipe 21 点手部追踪（最多 4 只手）+ 五指伸展 LED 状态与工业手势分类（`OPEN_PALM`、`FIST_LOCK`、`PINCH_GRIP`、`VICTORY_V` 等）。
  - 两个模块拥有各自独立的机械开关，关闭任一模块即跳过对应推理以节省算力。
- **外部网络流畅展示优化**：
  - **最新帧覆盖队列（Latest-Frame-Only WebSocket）**：后端自动丢弃积压旧帧，杜绝外网抖动造成的延迟累积。
  - **外网自适应码率（AUTO / WAN LITE / LAN HQ）**：实时监测网络 RTT，高延迟时自动降至 `320px–400px` 低体积 JPEG（约 `10–18 KB/帧`）。
  - **EMA 时域关键点平滑**：前端 60fps 平滑插值，外网低帧率推理下骨骼 HUD 依然丝滑无抖动。
  - **零外链字体阻塞 + GZip 压缩 + WS 断线自动重连 + 移动端前后摄像头切换**。

## 部署与启动

### 1. Windows 本地 / 局域网启动
双击 [start.bat](file:///d:/Agents/BM工作室/人体骨骼识别/start.bat)，或在终端执行：
```powershell
.\.venv\Scripts\python.exe server.py --host 0.0.0.0 --port 8000 --device auto
```

### 2. Linux 服务器一键部署（自动识别 GPU/CPU）
脚本会自动检测 `nvidia-smi`，有 NVIDIA 显卡则安装 PyTorch CUDA，无显卡则安装轻量 PyTorch CPU：
```bash
chmod +x deploy/deploy_linux.sh
./deploy/deploy_linux.sh
```

### 3. Docker 部署（支持 CPU 与 GPU 服务器）
```bash
# CPU 服务器
docker compose --profile cpu up -d --build

# NVIDIA GPU 服务器（需安装 nvidia-container-toolkit）
docker compose --profile gpu up -d --build
```

### 4. 外网浏览器调用摄像头说明（HTTPS）
现代浏览器安全策略要求：除 `localhost` 外，外部设备使用实时摄像头必须通过 **HTTPS** 访问（图像解析、视频解析、虚拟机械人演示可直接通过 HTTP 使用）。
- 方式 A：使用 Nginx 反向代理配置 HTTPS，参考 [deploy/nginx.conf.example](file:///d:/Agents/BM工作室/人体骨骼识别/deploy/nginx.conf.example)。
- 方式 B：直接向 `server.py` 传入证书启动：
  ```bash
  python server.py --host 0.0.0.0 --port 8443 --ssl-cert fullchain.pem --ssl-key privkey.pem
  ```
