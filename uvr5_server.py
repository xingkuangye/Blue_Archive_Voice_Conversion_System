"""
UVR5 独立 API 服务器 — 人声分离 / 混响消除

用法:
    pip install fastapi uvicorn python-multipart librosa soundfile numpy torch
    python uvr5_server.py

    可选参数:
    python uvr5_server.py --port 7861
    python uvr5_server.py --host 0.0.0.0 --port 8080

端点:
    POST /api/uvr5/separate  上传音频 → 返回人声 + 背景音
    POST /api/uvr5/dereverb  上传音频 → 返回干声 + 混响成分
    GET  /api/uvr5/health    健康检查
    GET  /api/download/{文件名}  下载处理结果
"""

import os, sys, uuid, json, logging, argparse, tempfile
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# ─── 日志 ───
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uvr5-server")

# ─── 全局 ───
# 检测 GPU
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"✅ 检测到 GPU: {gpu_name}")
        # 覆写 uvr5 模块的设备为 GPU
        import uvr5
        uvr5._device = torch.device("cuda")
        logger.info("✅ uvr5 已切换到 GPU 推理")
    else:
        logger.info("ℹ️ 使用 CPU 推理")
except Exception:
    logger.info("ℹ️ 使用 CPU 推理")

TEMP_DIR = Path(tempfile.gettempdir()) / "uvr5_server"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="UVR5 API", version="1.0.0", description="人声分离 / 混响消除 独立服务")


# ════════════════════════════════════════════
# 工具
# ════════════════════════════════════════════

def _save_audio(sr, arr, prefix):
    """保存音频到临时文件，返回路径"""
    import soundfile as sf
    import numpy as np
    name = f"{prefix}_{uuid.uuid4().hex[:8]}.wav"
    path = str(TEMP_DIR / name)
    # 确保数组形状正确: (samples,) 或 (samples, channels)
    if arr.ndim == 2 and arr.shape[0] <= 2:
        arr = arr.T
    # 确保 dtype 是 float64/32 或 int16
    if arr.dtype not in (np.float32, np.float64, np.int16, np.int32):
        arr = arr.astype(np.float32)
    try:
        sf.write(path, arr, sr)
    except Exception as e:
        # 回退：用 scipy 或手动构造 WAV
        logger.warning(f"soundfile 写入失败 ({e}), 尝试 scipy...")
        try:
            from scipy.io import wavfile
            # 归一化到 int16
            peak = np.abs(arr).max()
            if peak > 0:
                arr_int16 = (arr / peak * 32767).astype(np.int16)
            else:
                arr_int16 = arr.astype(np.int16)
            wavfile.write(path, sr, arr_int16)
        except Exception:
            # 最后回退：直接写 PCM 数据
            import struct
            peak = np.abs(arr).max()
            if peak > 0:
                arr_int16 = (arr / peak * 32767).astype(np.int16)
            else:
                arr_int16 = arr.astype(np.int16)
            with open(path, "wb") as f:
                n_channels = 1 if arr_int16.ndim == 1 else arr_int16.shape[1]
                n_samples = len(arr_int16) if arr_int16.ndim == 1 else arr_int16.shape[0]
                data_size = n_samples * n_channels * 2
                # WAV header
                f.write(b"RIFF")
                f.write(struct.pack("<I", 36 + data_size))
                f.write(b"WAVE")
                f.write(b"fmt ")
                f.write(struct.pack("<I", 16))
                f.write(struct.pack("<H", 1))  # PCM
                f.write(struct.pack("<H", n_channels))
                f.write(struct.pack("<I", sr))
                f.write(struct.pack("<I", sr * n_channels * 2))
                f.write(struct.pack("<H", n_channels * 2))
                f.write(struct.pack("<H", 16))
                f.write(b"data")
                f.write(struct.pack("<I", data_size))
                f.write(arr_int16.tobytes())
    return path


ALLOWED_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma")


def _allowed(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


# ════════════════════════════════════════════
# 端点
# ════════════════════════════════════════════

@app.get("/api/uvr5/health")
async def health():
    """健康检查"""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/uvr5/separate")
async def separate(
    audio: UploadFile = File(...),
    model_name: str = Form("mel_band_roformer"),
):
    """
    人声分离

    上传音频 → 返回人声 (vocals) 和背景音 (instrumental) 的下载路径

    - model_name: mel_band_roformer | htdemucs | htdemucs_ft
    """
    if not _allowed(audio.filename):
        raise HTTPException(400, f"不支持格式: {audio.filename}")

    # 保存上传
    ext = os.path.splitext(audio.filename)[1] or ".wav"
    input_path = TEMP_DIR / f"upload_{uuid.uuid4().hex[:8]}{ext}"
    content = await audio.read()
    if not content:
        raise HTTPException(400, "文件为空")
    with open(input_path, "wb") as f:
        f.write(content)

    try:
        from uvr5 import separate_audio_vocals
        logger.info(f"分离: {input_path} model={model_name}")
        vocals, inst, status = separate_audio_vocals(str(input_path), model_name)
        if vocals is None:
            raise RuntimeError(status)

        sr_v, arr_v = vocals
        sr_i, arr_i = inst
        vocals_path = _save_audio(sr_v, arr_v, "vocals")
        inst_path = _save_audio(sr_i, arr_i, "inst")

        return {"vocals": vocals_path, "instrumental": inst_path, "status": status}

    except Exception as e:
        logger.error(f"分离失败: {e}")
        raise HTTPException(500, str(e))
    finally:
        if input_path.exists():
            input_path.unlink()
        try:
            from vocal_roformer import unload_model as unload_vocal
            unload_vocal()
        except Exception:
            pass
        try:
            from demucs import pretrained
            import gc, torch
            pretrained._MODELS = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


@app.post("/api/uvr5/dereverb")
async def dereverb(
    audio: UploadFile = File(...),
    overlap: int = Form(4),
):
    """
    混响消除

    上传音频 → 返回干声 (dry) 和混响成分 (reverb) 的下载路径
    """
    if not _allowed(audio.filename):
        raise HTTPException(400, f"不支持格式: {audio.filename}")
    if overlap < 2 or overlap > 8:
        raise HTTPException(400, "overlap 必须在 2-8")

    ext = os.path.splitext(audio.filename)[1] or ".wav"
    input_path = TEMP_DIR / f"upload_{uuid.uuid4().hex[:8]}{ext}"
    content = await audio.read()
    if not content:
        raise HTTPException(400, "文件为空")
    with open(input_path, "wb") as f:
        f.write(content)

    try:
        from uvr5 import separate_dereverb
        logger.info(f"去混响: {input_path} overlap={overlap}")
        dry, reverb, status = separate_dereverb(str(input_path), overlap)
        if dry is None:
            raise RuntimeError(status)

        sr_d, arr_d = dry
        sr_r, arr_r = reverb
        dry_path = _save_audio(sr_d, arr_d, "dry")
        reverb_path = _save_audio(sr_r, arr_r, "reverb")

        return {"dry": dry_path, "reverb": reverb_path, "status": status}

    except Exception as e:
        logger.error(f"去混响失败: {e}")
        raise HTTPException(500, str(e))
    finally:
        if input_path.exists():
            input_path.unlink()
        try:
            from mdx23c_dereverb import unload_model as unload_mdx
            unload_mdx()
        except Exception:
            pass


@app.get("/api/download/{filename}")
async def download(filename: str):
    """下载处理结果音频"""
    fp = TEMP_DIR / filename
    if not fp.exists():
        raise HTTPException(404, "文件未找到")
    ext = os.path.splitext(filename)[1].lower()
    mime = {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
        ".flac": "audio/flac", ".ogg": "audio/ogg",
    }
    return FileResponse(str(fp), media_type=mime.get(ext, "application/octet-stream"))


# ════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="UVR5 独立 API 服务器")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7861, help="监听端口 (默认 7861)")
    parser.add_argument("--reload", action="store_true", help="热重载模式")
    args = parser.parse_args()

    logger.info(f"UVR5 API 启动: http://{args.host}:{args.port}")
    logger.info(f"API 文档: http://{args.host}:{args.port}/docs")
    uvicorn.run("uvr5_server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
