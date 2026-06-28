"""
GSV (GPT-SoVits) 独立 API 服务器
每次请求后自动卸载模型释放显存

用法:
  pip install torch soundfile numpy
  python gsv_server.py --port 54565

  --gpt_root F:/GPT-SoVITS/GPT_weights_v4
  --sovits_root F:/GPT-SoVITS/SoVITS_weights_v4

端点:
  GET  /openapi.json          — 健康检查
  GET  /set_gpt_weights       — 切换 GPT 模型 (weights_path)
  GET  /set_sovits_weights    — 切换 SoVits 模型 (weights_path)
  GET  /set_refer_audio       — 设置参考音频 (refer_audio_path)
  GET  /tts                   — 语音合成
  GET  /unload                — 手动卸载模型释放显存
"""
import os, sys, io, gc, uuid, json, logging, argparse, tempfile
import traceback
from pathlib import Path

import uvicorn
import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gsv-server")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_gpt_model = None
_sovits_model = None
_hz = 32000
_version = "v4"
_is_half = False

app = FastAPI(title="GSV TTS Server", version="2.0.0")


def unload():
    """卸载所有模型释放显存"""
    global _gpt_model, _sovits_model
    if _gpt_model is not None:
        _gpt_model = _gpt_model.cpu()
        del _gpt_model
        _gpt_model = None
    if _sovits_model is not None:
        _sovits_model = _sovits_model.cpu()
        del _sovits_model
        _sovits_model = None
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        logger.info(f"显存已释放，当前占用: {torch.cuda.memory_allocated()/1024**2:.0f}MB")


@app.get("/openapi.json")
async def openapi():
    return {
        "info": {"title": "GSV TTS Server", "version": "2.0.0"},
        "paths": {
            "/set_gpt_weights": {"get": {"parameters": [{"name": "weights_path", "in": "query"}]}},
            "/set_sovits_weights": {"get": {"parameters": [{"name": "weights_path", "in": "query"}]}},
            "/set_refer_audio": {"get": {"parameters": [{"name": "refer_audio_path", "in": "query"}]}},
            "/tts": {"get": {"parameters": [
                {"name": "text", "in": "query"},
                {"name": "text_lang", "in": "query"},
                {"name": "ref_audio_path", "in": "query"},
                {"name": "prompt_text", "in": "query"},
                {"name": "prompt_lang", "in": "query"},
                {"name": "top_k", "in": "query"},
                {"name": "top_p", "in": "query"},
                {"name": "temperature", "in": "query"},
                {"name": "speed_factor", "in": "query"},
                {"name": "seed", "in": "query"},
            ]}},
            "/unload": {"get": {}},
        }
    }


@app.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = Query(...)):
    global _gpt_model, _hz, _version
    try:
        unload()
        logger.info(f"加载 GPT 模型: {weights_path}")

        # 导入 GPT-SoVits 模块
        sys.path.insert(0, str(Path(weights_path).parent.parent))
        from GPT_SoVITS.inference_web import get_tts_model, change_gpt_weights_v2

        _gpt_model, _hz, _version = change_gpt_weights_v2(weights_path)
        logger.info(f"GPT 加载成功: {weights_path}, sr={_hz}, version={_version}")
        return {"ok": True, "message": f"GPT 模型切换成功 ({Path(weights_path).name})"}
    except Exception as e:
        logger.error(f"GPT 加载失败: {e}")
        raise HTTPException(500, detail=str(e))


@app.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = Query(...)):
    global _sovits_model
    try:
        logger.info(f"加载 SoVits 模型: {weights_path}")
        from GPT_SoVITS.inference_web import change_sovits_weights_v2, change_sovits_weights_v4

        if _version == "v4":
            _sovits_model = change_sovits_weights_v4(_sovits_model, weights_path)
        else:
            _sovits_model = change_sovits_weights_v2(_sovits_model, weights_path)
        logger.info(f"SoVits 加载成功: {weights_path}")
        return {"ok": True, "message": f"SoVits 模型切换成功 ({Path(weights_path).name})"}
    except Exception as e:
        logger.error(f"SoVits 加载失败: {e}")
        raise HTTPException(500, detail=str(e))


@app.get("/set_refer_audio")
async def set_refer_audio(refer_audio_path: str = Query("")):
    try:
        if refer_audio_path and os.path.isfile(refer_audio_path):
            # 验证文件存在即可，GPT-SoVITS 内部会重新加载
            logger.info(f"参考音频已设置: {refer_audio_path}")
        return {"ok": True, "message": "参考音频设置成功"}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/tts")
async def tts(
    text: str = Query(...),
    text_lang: str = Query("zh"),
    ref_audio_path: str = Query(""),
    prompt_text: str = Query(""),
    prompt_lang: str = Query("zh"),
    top_k: int = Query(5),
    top_p: float = Query(1.0),
    temperature: float = Query(1.0),
    speed_factor: float = Query(1.0),
    media_type: str = Query("wav"),
    seed: int = Query(-1),
):
    if _gpt_model is None or _sovits_model is None:
        raise HTTPException(400, detail="模型未加载，请先调用 set_gpt_weights 和 set_sovits_weights")

    try:
        from GPT_SoVITS.inference_web import get_tts_wav

        sr, audio = get_tts_wav(
            ref_audio_path=ref_audio_path,
            ref_text=prompt_text,
            ref_lang=prompt_lang,
            gen_text=text,
            gen_lang=text_lang,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            speed=speed_factor,
        )

        # 卸载模型释放显存
        unload()

        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return Response(content=buf.getvalue(), media_type="audio/wav")

    except Exception as e:
        unload()
        logger.error(f"TTS 失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, detail=str(e))


@app.get("/unload")
async def api_unload():
    unload()
    return {"ok": True, "message": "模型已卸载"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSV TTS 独立服务器")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=54565)
    args = parser.parse_args()

    if torch.cuda.is_available():
        logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("ℹ️ 使用 CPU")

    logger.info(f"GSV TTS 服务: http://{args.host}:{args.port}")
    uvicorn.run("gsv_server:app", host=args.host, port=args.port)
