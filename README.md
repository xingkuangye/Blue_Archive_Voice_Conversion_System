# Blue Archive RVC — Voice Conversion System

基于 RVC (Retrieval-based Voice Conversion) 的 Blue Archive 角色变声系统，支持 GPU/CPU 推理，集成 GPT-SoVits TTS 远程调用。

## 功能

- **翻唱变声** — 从 RVC 角色模型中选择角色，上传音频或输入文字（Edge-TTS），转换声音
- **人声分离** — 使用 Mel-Band Roformer 或 Demucs 分离人声和背景音
- **混响消除** — 使用 MDX23C De-Reverb 模型消除音频混响
- **GSV TTS** — 远程调用 GPT-SoVits 服务进行高质量语音合成
- **远程 RVC** — 支持配置远程 GPU 服务器，自动回退到本地 CPU
- **管理员后台** (`/admin`) — 管理角色模型、GSV 配置、远程 RVC 配置

## 本地启动

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python3 run.py

# 访问 http://localhost:7860
```

## 目录结构

```
├── backend/
│   ├── api.py              # FastAPI 主服务
│   ├── gsv.py              # GSV 远程 TTS 集成
│   ├── rvc_remote.py       # 远程 RVC 调用
│   └── admin.py            # 后台管理逻辑
├── static/
│   ├── index.html           # 主页面
│   ├── admin.html           # 后台管理页面
│   ├── admin_login.html     # 管理员登录页
│   ├── css/style.css        # 样式
│   └── js/app.js            # 前端逻辑
├── weights/                 # 角色模型权重（需自行放置）
├── lib/infer_pack/          # RVC 模型架构
├── config.py                # 硬件配置
├── vc_infer_pipeline.py     # 变声管线
├── uvr5.py                  # 人声分离/去混响
└── run.py                   # 启动入口
```

## 管理员后台

访问 `http://localhost:7860/admin`

默认密码: `admin123`（可通过环境变量 `ADMIN_PASSWORD` 修改）

## 远程 RVC 服务器部署

参考 `rvc_server/api_server.py`，在有 GPU 的机器上启动远程 API 服务。

## License

MIT
