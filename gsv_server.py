"""
GSV TTS 独立服务器（适配 GPT-SoVITS v2pro api_v2.py）
每次 TTS 后自动卸载模型释放显存

用法:
  runtime\\python.exe gsv_server.py --port 54565

放在 GPT-SoVITS 根目录下运行
"""
import os, sys, gc, json, logging, argparse, traceback, signal, wave
from pathlib import Path
from io import BytesIO

import uvicorn
import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gsv-server")

# 把当前目录和 GPT_SoVITS 加入路径
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tts_pipeline = None
_tts_config = None

app = FastAPI(title="GSV TTS Server", version="2.0.0")


def unload():
    """销毁 pipeline 释放显存"""
    global _tts_pipeline, _tts_config
    if _tts_pipeline is not None:
        del _tts_pipeline
        _tts_pipeline = None
    if _tts_config is not None:
        del _tts_config
        _tts_config = None
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        logger.info(f"显存已释放，当前占用: {torch.cuda.memory_allocated()/1024**2:.0f}MB")
    else:
        logger.info("模型已卸载")


def get_pipeline():
    """获取或创建 TTS pipeline"""
    global _tts_pipeline, _tts_config
    if _tts_pipeline is None:
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        from tools.i18n.i18n import I18nAuto
        i18n = I18nAuto()
        config_path = "GPT_SoVITS/configs/tts_infer.yaml"
        if not os.path.isfile(config_path):
            config_path = "GPT-SoVITS/configs/tts_infer.yaml"
        _tts_config = TTS_Config(config_path)
        _tts_pipeline = TTS(_tts_config)
        logger.info("TTS pipeline 初始化完成")
    return _tts_pipeline


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


@app.get("/openapi.json")
async def openapi_schema():
    return {
        "openapi": "3.1.0",
        "info": {"title": "GSV TTS Server", "version": "2.0.0"},
        "paths": {
            "/set_gpt_weights": {"get": {"parameters": [{"name": "weights_path", "in": "query", "required": True}]}},
            "/set_sovits_weights": {"get": {"parameters": [{"name": "weights_path", "in": "query", "required": True}]}},
            "/set_refer_audio": {"get": {"parameters": [{"name": "refer_audio_path", "in": "query", "required": True}]}},
            "/tts": {"get": {"parameters": [
                {"name": "text", "in": "query", "required": True},
                {"name": "text_lang", "in": "query"},
                {"name": "ref_audio_path", "in": "query", "required": True},
                {"name": "prompt_text", "in": "query"},
                {"name": "prompt_lang", "in": "query"},
                {"name": "top_k", "in": "query"},
                {"name": "top_p", "in": "query"},
                {"name": "temperature", "in": "query"},
                {"name": "speed_factor", "in": "query"},
                {"name": "text_split_method", "in": "query"},
            ]}},
            "/unload": {"get": {}},
        }
    }


@app.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = Query(...)):
    try:
        pipeline = get_pipeline()
        pipeline.init_t2s_weights(weights_path)
        logger.info(f"GPT 模型加载成功: {Path(weights_path).name}")
        return JSONResponse({"message": "success"})
    except Exception as e:
        logger.error(f"GPT 加载失败: {e}")
        raise HTTPException(400, detail=str(e))


@app.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = Query(...)):
    try:
        pipeline = get_pipeline()
        pipeline.init_vits_weights(weights_path)
        logger.info(f"SoVits 模型加载成功: {Path(weights_path).name}")
        return JSONResponse({"message": "success"})
    except Exception as e:
        logger.error(f"SoVits 加载失败: {e}")
        raise HTTPException(400, detail=str(e))


@app.get("/set_refer_audio")
async def set_refer_audio(refer_audio_path: str = Query(...)):
    try:
        pipeline = get_pipeline()
        pipeline.set_ref_audio(refer_audio_path)
        logger.info(f"参考音频已设置: {refer_audio_path}")
        return JSONResponse({"message": "success"})
    except Exception as e:
        logger.error(f"参考音频设置失败: {e}")
        raise HTTPException(400, detail=str(e))


@app.get("/tts")
async def tts_endpoint(
    text: str = Query(...),
    text_lang: str = Query("zh"),
    ref_audio_path: str = Query(...),
    prompt_text: str = Query(""),
    prompt_lang: str = Query("zh"),
    top_k: int = Query(5),
    top_p: float = Query(1.0),
    temperature: float = Query(1.0),
    speed_factor: float = Query(1.0),
    text_split_method: str = Query("cut5"),
    media_type: str = Query("wav"),
    seed: int = Query(-1),
    batch_size: int = Query(1),
    parallel_infer: bool = Query(True),
    repetition_penalty: float = Query(1.35),
    sample_steps: int = Query(32),
    streaming_mode: bool = Query(False),
):
    try:
        pipeline = get_pipeline()

        req = {
            "text": text,
            "text_lang": text_lang.lower(),
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": [],
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang.lower(),
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": batch_size,
            "speed_factor": float(speed_factor),
            "split_bucket": True,
            "fragment_interval": 0.3,
            "seed": seed,
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": parallel_infer,
            "repetition_penalty": repetition_penalty,
            "sample_steps": sample_steps,
            "super_sampling": False,
        }

        _, audio_data = next(pipeline.run(req))
        audio_data = pack_wav(BytesIO(), audio_data, 32000).getvalue()

        # TTS 完成后卸载模型释放显存
        unload()

        return Response(audio_data, media_type="audio/wav")

    except Exception as e:
        unload()
        logger.error(f"TTS 失败: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})


@app.get("/unload")
async def api_unload():
    unload()
    return {"ok": True, "message": "已卸载"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=54565)
    args = parser.parse_args()

    if torch.cuda.is_available():
        logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("ℹ️ 使用 CPU 推理")

    logger.info(f"GSV TTS: http://{args.host}:{args.port}")
    uvicorn.run("gsv_server:app", host=args.host, port=args.port, workers=1)
