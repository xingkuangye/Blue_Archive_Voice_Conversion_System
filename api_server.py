#!/usr/bin/env python3
"""
RVC 远程 API 服务器
支持 GPU (CUDA) 推理，用完即卸载模型释放显存
支持文件名搜索：本地传 model_file/index_file 文件名，远程自动在 weights/ 下搜索
"""
import os, sys, json, uuid, time, logging, threading, traceback
from pathlib import Path
from contextlib import asynccontextmanager

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", "7860"))
HOST = os.environ.get("HOST", "0.0.0.0")
API_KEY = os.environ.get("RVC_API_KEY", "")

categories_meta = []
_queue = []
_status = {}
_processing = False
_lock = threading.Lock()


# ═══════ 文件搜索 ═══════

def _find_weights_file(filename):
    """在 weights/ 下递归搜索文件，返回完整路径"""
    if not filename:
        return ""
    for root, dirs, files in os.walk("weights"):
        if filename in files:
            return os.path.join(root, filename)
    return ""


# ═══════ 队列 ═══════

def enqueue(params: dict) -> str:
    global _processing
    qid = uuid.uuid4().hex[:12]
    with _lock:
        _queue.append((qid, params))
        _status[qid] = {"status": "queued", "position": len(_queue)}
        if not _processing:
            _processing = True
            t = threading.Thread(target=_worker, daemon=True)
            t.start()
    return qid

def get_status(qid: str) -> dict:
    with _lock:
        s = _status.get(qid)
    return dict(s) if s else {"status": "unknown"}

def _worker():
    global _processing
    while True:
        with _lock:
            if not _queue:
                _processing = False; return
            qid, params = _queue.pop(0)
            _status[qid] = {"status": "processing", "progress": 0}
        try:
            logger.info(f"[队列] 开始 {qid}")
            result = _do_convert(params)
            result["status"] = "done"
            with _lock: _status[qid] = result
            logger.info(f"[队列] 完成 {qid}")
        except Exception as e:
            logger.error(f"[队列] 失败 {qid}: {e}\n{traceback.format_exc()}")
            with _lock: _status[qid] = {"status": "error", "error": str(e)}


# ═══════ 模型加载 + 用完即卸载 ═══════

def _load_hubert(cfg):
    import torch
    from fairseq import checkpoint_utils
    from fairseq.data.dictionary import Dictionary
    torch.serialization.add_safe_globals([Dictionary])
    logger.info("加载 HuBERT...")
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"], suffix="")
    hubert = models[0].to(cfg.device)
    hubert = hubert.half() if cfg.is_half else hubert.float()
    hubert.eval()
    logger.info("HuBERT 加载完成")
    return hubert


def _get_model(character, cfg, model_file="", index_file=""):
    import torch, json
    from lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono,
    )
    from vc_infer_pipeline import VC

    global categories_meta
    model_path = ""
    index_path = ""

    if model_file:
        # 优先用文件名搜索（无需 folder_info.json 同步）
        model_path = _find_weights_file(model_file)
        if index_file:
            index_path = _find_weights_file(index_file)
    else:
        # 回退：通过 folder_info.json 查找（兼容旧版）
        if not categories_meta:
            _load_categories()
        char_info = None
        for cat in categories_meta:
            for ch in cat["characters"]:
                if ch["name"] == character: char_info = ch; break
        if char_info is None:
            raise ValueError(f"角色 '{character}' 未找到")
        with open("weights/folder_info.json", encoding="utf-8") as f:
            folder_info = json.load(f)
        for cat_name, cat_info in folder_info.items():
            if cat_info["title"] == char_info["category"]:
                folder = cat_info["folder_path"]
                with open(f"weights/{folder}/model_info.json", encoding="utf-8") as f:
                    model_info = json.load(f)
                for cname, minfo in model_info.items():
                    if cname == character or minfo.get("title") == char_info["title"]:
                        model_path = f"weights/{folder}/{cname}/{minfo['model_path']}"
                        index_path = f"weights/{folder}/{cname}/{minfo['feature_retrieval_library']}"
                        break

    if not model_path or not os.path.isfile(model_path):
        raise ValueError(f"模型文件未找到: {model_path or model_file}")

    logger.info(f"加载角色模型: {character} -> {model_path}")
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    if version == "v1":
        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=cfg.is_half) if if_f0 == 1 else SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    else:
        net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=cfg.is_half) if if_f0 == 1 else SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(cfg.device)
    net_g = net_g.half() if cfg.is_half else net_g.float()
    vc = VC(tgt_sr, cfg)

    return {"net_g": net_g, "vc": vc, "tgt_sr": tgt_sr, "if_f0": if_f0, "version": version, "index_path": index_path}


def _unload_models(hubert=None, mc=None):
    """用完立即销毁模型对象并清空显存"""
    import torch, gc
    logger.info("卸载模型，释放显存...")
    if mc:
        refs = [mc["net_g"], mc["vc"]]
        for r in refs:
            if hasattr(r, 'cpu'):
                try: r.cpu()
                except: pass
            del r
    if hubert:
        try: hubert.cpu()
        except: pass
        del hubert
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("显存已释放")


def _do_convert(params: dict) -> dict:
    import torch, librosa, numpy as np, soundfile as sf
    from config import Config

    character = params["character"]
    audio_path = params["audio_path"]
    model_file = params.get("model_file", "")
    index_file = params.get("index_file", "")
    f0_up_key = int(params["f0_up_key"])
    f0_method = params["f0_method"]
    index_rate = params["index_rate"]
    filter_radius = params["filter_radius"]
    resample_sr = params["resample_sr"]
    rms_mix_rate = params["rms_mix_rate"]
    protect = params["protect"]

    cfg = Config()
    hubert = None
    mc = None
    try:
        hubert = _load_hubert(cfg)
        mc = _get_model(character, cfg, model_file, index_file)
        net_g, vc = mc["net_g"], mc["vc"]
        tgt_sr, if_f0 = mc["tgt_sr"], mc["if_f0"]
        version, index_path = mc["version"], mc["index_path"]

        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        times = [0, 0, 0]

        audio_opt = vc.pipeline(
            hubert, net_g, 0, audio, audio_path,
            times, f0_up_key, f0_method, index_path, index_rate,
            if_f0, filter_radius, tgt_sr, resample_sr,
            rms_mix_rate, version, protect, f0_file=None,
        )

        output_path = f"temp/output_{uuid.uuid4().hex[:8]}.wav"
        sf.write(output_path, audio_opt, tgt_sr)

        from datetime import datetime
        info = (f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"npy: {times[0]:.2f}s | f0: {times[1]:.2f}s | infer: {times[2]:.2f}s | device: {cfg.device}")

        return {"output_path": output_path, "sample_rate": tgt_sr, "info": info}
    finally:
        # 不管成功还是失败，用完立即卸载
        _unload_models(hubert, mc)


# ═══════ 辅助 ═══════

def _load_categories():
    global categories_meta; categories_meta = []
    if not os.path.isfile("weights/folder_info.json"): return
    with open("weights/folder_info.json", encoding="utf-8") as f:
        folder_info = json.load(f)
    for _, ci in folder_info.items():
        if not ci.get("enable", True): continue
        fp = ci["folder_path"]
        mi_path = f"weights/{fp}/model_info.json"
        if not os.path.isfile(mi_path): continue
        with open(mi_path, encoding="utf-8") as f:
            models_info = json.load(f)
        cl = []
        for cn, info in models_info.items():
            if not info.get("enable", True): continue
            cv = f"weights/{fp}/{cn}/{info['cover']}"
            cl.append({"name": cn, "title": info["title"], "author": info.get("author",""),
                       "cover": cv if os.path.isfile(cv) else None, "version": "v2", "category": ci["title"]})
        categories_meta.append({"title": ci["title"], "folder": fp, "description": ci.get("description",""), "characters": cl})


def _check_api_key(r: Request) -> bool:
    return True if not API_KEY else r.headers.get("X-Api-Key", "") == API_KEY


def check_gpu():
    import torch
    if torch.cuda.is_available():
        logger.info(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA 不可用，将使用 CPU 推理")


# ═══════ FastAPI ═══════

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_categories()
    logger.info(f"已加载 {sum(len(c['characters']) for c in categories_meta)} 个角色")
    check_gpu()
    yield

app = FastAPI(title="RVC Remote API", version="1.0.0", lifespan=lifespan)
os.makedirs("temp", exist_ok=True)


@app.get("/api/health")
async def health():
    import torch
    return {
        "status": "ok", "ml_available": True,
        "cuda": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

@app.get("/api/models")
async def list_models():
    _load_categories(); return {"categories": categories_meta}

@app.post("/api/upload")
async def upload_audio(request: Request, file: UploadFile = File(...)):
    if not _check_api_key(request): raise HTTPException(403)
    ext = os.path.splitext(file.filename)[1] or ".wav"
    d = f"temp/upload_{os.urandom(4).hex()}{ext}"
    with open(d, "wb") as f: f.write(await file.read())
    return {"path": d, "filename": file.filename}

@app.post("/api/convert")
async def convert_voice(
    request: Request, character: str = Form(...), audio_path: str = Form(...),
    model_file: str = Form(""), index_file: str = Form(""),
    f0_up_key: int = Form(0), f0_method: str = Form("pm"),
    index_rate: float = Form(0.7), filter_radius: int = Form(3),
    resample_sr: int = Form(0), rms_mix_rate: float = Form(1.0),
    protect: float = Form(0.5),
):
    if not _check_api_key(request): raise HTTPException(403)
    if not os.path.isfile(audio_path): raise HTTPException(400, detail="音频文件未找到")
    qid = enqueue({"character": character, "audio_path": audio_path,
        "model_file": model_file, "index_file": index_file,
        "f0_up_key": f0_up_key, "f0_method": f0_method,
        "index_rate": index_rate, "filter_radius": filter_radius,
        "resample_sr": resample_sr, "rms_mix_rate": rms_mix_rate, "protect": protect})
    return {"queue_id": qid}

@app.get("/api/queue/{queue_id}")
async def queue_status(request: Request, queue_id: str):
    if not _check_api_key(request): raise HTTPException(403)
    return get_status(queue_id)

@app.post("/api/control")
async def control(request: Request, command: str = Form(...)):
    if not _check_api_key(request): raise HTTPException(403)
    if command == "restart":
        os.execl(sys.executable, sys.executable, *sys.argv)
    elif command == "clear_queue":
        with _lock: _queue.clear(); _status.clear(); global _processing; _processing = False
        return {"ok": True, "message": "队列已清空"}
    raise HTTPException(400, detail=f"未知命令: {command}")

@app.get("/api/download/{filename:path}")
async def download(request: Request, filename: str):
    if not _check_api_key(request): raise HTTPException(403)
    fp = f"temp/{filename}"
    if not os.path.isfile(fp): raise HTTPException(404)
    return FileResponse(fp, media_type="audio/wav", filename=filename)


if __name__ == "__main__":
    uvicorn.run("api_server:app", host=HOST, port=PORT, reload=False, log_level="info")
