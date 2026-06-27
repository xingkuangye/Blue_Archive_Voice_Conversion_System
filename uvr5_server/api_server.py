"""
UVR5 独立 API 服务器
人声分离 + 混响消除

用法:
    python uvr5_server/api_server.py

    或指定端口:
    python uvr5_server/api_server.py --port 7861

端点:
    POST /api/uvr5/separate   — 人声分离（上传音频，返回人声 + 背景音）
    POST /api/uvr5/dereverb   — 混响消除（上传音频，返回干声 + 混响成分）
    GET  /api/uvr5/health     — 健康检查
"""
import os
import sys
import uuid
import json
import logging
import argparse
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, Response

# 确保项目根目录在路径中（使 from uvr5 import ... 可用）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("uvr5-server")

# ─── 全局状态 ───
_models_loaded = False
_temp_dir = ROOT / "temp"
_temp_dir.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models_loaded
    logger.info("UVR5 API 服务器启动")
    # 提前导入 uvr5 模块以检查环境
    try:
        import uvr5
        _models_loaded = True
        logger.info("uvr5 模块加载成功")
    except ImportError as e:
        logger.warning(f"uvr5 模块加载失败，部分功能可能不可用: {e}")
        _models_loaded = False
    yield
    logger.info("UVR5 API 服务器关闭")


app = FastAPI(
    title="UVR5 API Server",
    version="1.0.0",
    description="UVR5 人声分离 / 混响消除 独立 API",
    lifespan=lifespan,
)


# ════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════

def _save_audio(data: tuple, prefix: str) -> str:
    """
    保存 (sample_rate, numpy_array) 到临时文件
    返回相对路径 (temp/filename.wav)
    """
    import soundfile as sf
    sr, arr = data
    name = f"{prefix}_{uuid.uuid4().hex[:8]}.wav"
    path = str(_temp_dir / name)
    sf.write(path, arr, sr)
    return path


def _allowed_audio(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma")


# ════════════════════════════════════════════
# API 端点
# ════════════════════════════════════════════

@app.get("/api/uvr5/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": _models_loaded,
        "version": "1.0.0",
    }


@app.post("/api/uvr5/separate")
async def separate(
    audio: UploadFile = File(...),
    model_name: str = Form("mel_band_roformer"),
):
    """
    人声分离
    - audio: 音频文件 (wav/mp3/m4a/flac/ogg)
    - model_name: mel_band_roformer (默认) | htdemucs | htdemucs_ft
    返回: { vocals, instrumental, status } 三个音频文件路径
    """
    if not _models_loaded:
        raise HTTPException(status_code=503, detail="UVR5 模块未加载，请检查依赖")

    if not _allowed_audio(audio.filename):
        raise HTTPException(status_code=400, detail=f"不支持的音频格式: {audio.filename}")

    # 保存上传文件到临时目录
    ext = os.path.splitext(audio.filename)[1] or ".wav"
    input_path = _temp_dir / f"upload_{uuid.uuid4().hex[:8]}{ext}"
    try:
        content = await audio.read()
        if not content:
            raise HTTPException(status_code=400, detail="上传的音频文件为空")
        with open(input_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    try:
        from uvr5 import separate_audio_vocals

        logger.info(f"分离: {input_path} model={model_name}")
        vocals, inst, status = separate_audio_vocals(str(input_path), model_name)

        if vocals is None or inst is None:
            raise RuntimeError(status)

        vocals_path = _save_audio(vocals, "vocals")
        inst_path = _save_audio(inst, "inst")

        return {
            "vocals": vocals_path,
            "instrumental": inst_path,
            "status": status,
        }
    except Exception as e:
        logger.error(f"分离失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理上传的临时文件
        if input_path.exists():
            input_path.unlink(missing_ok=True)


@app.post("/api/uvr5/dereverb")
async def dereverb(
    audio: UploadFile = File(...),
    overlap: int = Form(4),
):
    """
    混响消除
    - audio: 音频文件
    - overlap: 重叠倍数 (2-8, 默认 4)
    返回: { dry, reverb, status } 两个音频文件路径
    """
    if not _models_loaded:
        raise HTTPException(status_code=503, detail="UVR5 模块未加载，请检查依赖")

    if not _allowed_audio(audio.filename):
        raise HTTPException(status_code=400, detail=f"不支持的音频格式: {audio.filename}")

    if overlap < 2 or overlap > 8:
        raise HTTPException(status_code=400, detail="overlap 必须在 2-8 之间")

    ext = os.path.splitext(audio.filename)[1] or ".wav"
    input_path = _temp_dir / f"upload_{uuid.uuid4().hex[:8]}{ext}"
    try:
        content = await audio.read()
        if not content:
            raise HTTPException(status_code=400, detail="上传的音频文件为空")
        with open(input_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    try:
        from uvr5 import separate_dereverb

        logger.info(f"去混响: {input_path} overlap={overlap}")
        dry, reverb, status = separate_dereverb(str(input_path), overlap)

        if dry is None or reverb is None:
            raise RuntimeError(status)

        dry_path = _save_audio(dry, "dry")
        reverb_path = _save_audio(reverb, "reverb")

        return {
            "dry": dry_path,
            "reverb": reverb_path,
            "status": status,
        }
    except Exception as e:
        logger.error(f"去混响失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if input_path.exists():
            input_path.unlink(missing_ok=True)


# ════════════════════════════════════════════
# 文件下载
# ════════════════════════════════════════════

@app.get("/api/download/{filename:path}")
async def download(filename: str):
    """下载临时文件"""
    fp = _temp_dir / filename
    if not fp.exists() or not fp.is_file():
        raise HTTPException(status_code=404, detail="文件未找到")
    return FileResponse(str(fp))


# ════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════

def run():
    parser = argparse.ArgumentParser(description="UVR5 API Server")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7861, help="监听端口 (默认 7861)")
    parser.add_argument("--reload", action="store_true", help="热重载模式")
    args = parser.parse_args()

    logger.info(f"UVR5 API Server 启动: http://{args.host}:{args.port}")
    logger.info(f"API 文档: http://{args.host}:{args.port}/docs")
    uvicorn.run(
        "uvr5_server.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    run()
