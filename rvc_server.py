"""
RVC 独立推理服务器（GPU）
本地只需传角色名 + 模型文件名，远程自动在 weights/ 下搜索

用法:
  python rvc_server.py --port 7862
  需有 weights/ hubert_base.pt rmvpe.pt lib/ config.py vc_infer_pipeline.py
"""
import os, sys, gc, json, uuid, time, logging, argparse, traceback
from pathlib import Path
import uvicorn
import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import Response, FileResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rvc-server")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="RVC Server", version="1.0.0")


def find_model_file(character, filename):
    """在 weights/ 下递归搜索文件"""
    if not filename:
        return None
    weights_dir = Path("weights")
    if not weights_dir.exists():
        return None
    for p in weights_dir.rglob(filename):
        if p.is_file():
            return str(p)
    # 再搜索角色名目录
    for p in weights_dir.rglob(f"*/{character}/*"):
        if p.is_file() and p.name == filename:
            return str(p)
    return None


@app.get("/api/rvc/health")
async def health():
    return {"status": "ok", "device": str(_device)}


@app.post("/api/rvc/convert")
async def rvc_convert(
    audio: UploadFile = File(...),
    character: str = Form(...),
    model_file: str = Form(...),
    index_file: str = Form(""),
    f0_up_key: int = Form(0),
    f0_method: str = Form("rmvpe"),
    index_rate: float = Form(0.7),
    filter_radius: int = Form(3),
    resample_sr: int = Form(0),
    rms_mix_rate: float = Form(1.0),
    protect: float = Form(0.5),
):
    input_path = f"temp/input_{uuid.uuid4().hex[:8]}.wav"
    os.makedirs("temp", exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(await audio.read())

    try:
        # 搜索模型文件
        model_path = find_model_file(character, model_file)
        if not model_path:
            raise HTTPException(404, f"模型文件 {model_file} 未找到")

        index_path = ""
        if index_file:
            index_path = find_model_file(character, index_file)

        logger.info(f"转换: {character} model={model_path}")

        # 注入路径 → 原有 task_convert 逻辑
        import librosa, soundfile as sf2
        from fairseq import checkpoint_utils
        from fairseq.data.dictionary import Dictionary
        from lib.infer_pack.models import (SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono,
                                            SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono)
        from config import Config
        from vc_infer_pipeline import VC

        cfg = Config()
        torch.serialization.add_safe_globals([Dictionary])
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"], suffix="")
        hubert = models[0].to(cfg.device)
        hubert = hubert.half() if cfg.is_half else hubert.float()
        hubert.eval()

        cpt = torch.load(model_path, map_location="cpu")
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")

        if version == "v1":
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=cfg.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=cfg.is_half)
        del net_g.enc_q
        net_g.load_state_dict(cpt["weight"], strict=False)
        net_g.to(cfg.device).eval()
        if cfg.is_half: net_g = net_g.half()

        vc = VC(tgt_sr, cfg)

        audio_np, sr = librosa.load(input_path, sr=16000, mono=True)
        audio_opt = vc.pipeline(
            (hubert, net_g), 0, audio_np, input_path,
            f0_up_key, f0_method, index_path, index_rate,
            if_f0, filter_radius, tgt_sr, resample_sr, rms_mix_rate, protect,
        )

        output_path = f"temp/output_{uuid.uuid4().hex[:8]}.wav"
        sf2.write(output_path, audio_opt, tgt_sr)

        # 清理显存
        del hubert, net_g, vc
        gc.collect()
        torch.cuda.empty_cache()

        return FileResponse(output_path, media_type="audio/wav", filename=f"{character}.wav")

    except Exception as e:
        logger.error(f"转换失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, detail=str(e))
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    logger.info(f"RVC Server: http://0.0.0.0:{args.port}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    uvicorn.run("rvc_server:app", host="0.0.0.0", port=args.port)
