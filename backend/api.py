"""
FastAPI 后端 — Blue Archive RVC
统一异步队列 + 进度上报
"""
import os, json, uuid, time, logging, threading, traceback
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from PIL import Image

# Admin password (override via env var)
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── 全局 ───
ml_available = False
categories_meta = []



# ════════════════════════════════════════════
# 通用任务队列
# ════════════════════════════════════════════

class TaskQueue:
    """
    单线程串行任务队列，支持进度上报
    每个任务由 (task_type, params, fn) 组成
    fn 接受 (params, progress_callback) 并返回 dict
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._queue = []       # [(qid, task_type, params, fn)]
        self._status = {}      # {qid: {...}}
        self._processing = False
        self._thread = None
        self._avg_speed = 0.8

    def enqueue(self, task_type: str, params: dict, fn) -> str:
        qid = uuid.uuid4().hex[:12]
        with self._lock:
            self._queue.append((qid, task_type, params, fn))
            self._status[qid] = {"status": "queued", "position": len(self._queue)}
            self._refresh_positions()
            if not self._processing:
                self._processing = True
                self._thread = threading.Thread(target=self._worker, daemon=True)
                self._thread.start()
        return qid

    def set_progress(self, qid: str, progress: int, message: str = ""):
        """由工作线程调用更新进度"""
        with self._lock:
            if qid in self._status and self._status[qid].get("status") == "processing":
                self._status[qid]["progress"] = min(progress, 99)
                if message:
                    self._status[qid]["message"] = message

    def get_status(self, qid: str) -> dict:
        with self._lock:
            s = self._status.get(qid)
        if s is None:
            return {"status": "unknown"}
        return dict(s)  # 返回副本

    def _refresh_positions(self):
        for i, (qid, *_) in enumerate(self._queue):
            self._status[qid] = {"status": "queued", "position": i + 1}

    def _estimate_wait(self) -> float:
        """前面所有任务估计耗时"""
        total = 0.0
        with self._lock:
            for _, _, p, _ in self._queue[:-1]:
                total += p.get("_duration_sec", 5) * self._avg_speed
        return total

    def _worker(self):
        while True:
            with self._lock:
                if not self._queue:
                    self._processing = False
                    return
                qid, ttype, params, fn = self._queue.pop(0)
                self._refresh_positions()
                self._status[qid] = {"status": "processing", "progress": 0, "message": "初始化..."}

            start = time.time()
            try:
                dur = params.get("_duration_sec", 5)
                logger.info(f"[队列] 开始 {qid} [{ttype}], 时长={dur:.1f}s")

                # 包装进度回调
                def cb(pct, msg=""):
                    self.set_progress(qid, pct, msg)

                result = fn(params, cb)
                elapsed = time.time() - start
                speed = elapsed / max(dur, 0.01)
                self._avg_speed = self._avg_speed * 0.7 + speed * 0.3
                logger.info(f"[队列] 完成 {qid}: {elapsed:.1f}s (speed={speed:.2f}x)")

                result["status"] = "done"
                result["progress"] = 100
                with self._lock:
                    self._status[qid] = result
            except Exception as e:
                logger.error(f"[队列] 失败 {qid}: {e}\n{traceback.format_exc()}")
                with self._lock:
                    self._status[qid] = {"status": "error", "error": str(e)}


queue = TaskQueue()


# ════════════════════════════════════════════
# 任务函数
# ════════════════════════════════════════════

def task_convert(params: dict, cb) -> dict:
    # Try remote first
    try:
        from backend.rvc_remote import load_config, convert_remote
        cfg = load_config()
        if cfg.get("enabled") and cfg.get("api_url"):
            cb(10, "远程处理中...")
            import asyncio, librosa, soundfile as sf, numpy as np, io
            audio_path = params["audio_path"]
            with open(audio_path, "rb") as af:
                audio_data = af.read()
            result_bytes = asyncio.run(convert_remote(params["character"], audio_data, params))
            output_path = f"temp/output_{uuid.uuid4().hex[:8]}.wav"
            with open(output_path, "wb") as f:
                f.write(result_bytes)
            sr = librosa.get_duration(path=output_path)
            import soundfile as sf2
            data, sr = sf2.read(output_path)
            cb(100, "远程处理完成")
            return {"output_path": output_path, "sample_rate": int(sr), "info": "远程 RVC 处理"}
    except Exception as e:
        logger.warning(f"远程 RVC 不可用，回退到本地: {e}")
        cb(5, "远程不可用，使用本地处理")

    import torch, librosa, numpy as np, soundfile as sf
    from fairseq import checkpoint_utils
    from fairseq.data.dictionary import Dictionary
    from lib.infer_pack.models import (SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono,
                                        SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono)
    from config import Config
    from vc_infer_pipeline import VC

    cb(5, "加载配置...")
    cfg = Config()

    character = params["character"]
    audio_path = params["audio_path"]
    f0_up_key = int(params["f0_up_key"])
    f0_method = params["f0_method"]
    index_rate = params["index_rate"]
    filter_radius = params["filter_radius"]
    resample_sr = params["resample_sr"]
    rms_mix_rate = params["rms_mix_rate"]
    protect = params["protect"]

    char_info = None
    for cat in categories_meta:
        for ch in cat["characters"]:
            if ch["name"] == character:
                char_info = ch; break
    if char_info is None:
        raise ValueError(f"角色 '{character}' 未找到")

    cb(10, "加载 HuBERT...")
    torch.serialization.add_safe_globals([Dictionary])
    hubert_models, _, _ = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"], suffix="")
    hubert = hubert_models[0].to(cfg.device)
    hubert = hubert.half() if cfg.is_half else hubert.float()
    hubert.eval()

    cb(20, "加载角色模型...")
    with open("weights/folder_info.json") as f:
        folder_info = json.load(f)
    for cat_name, cat_info in folder_info.items():
        if cat_info["title"] == char_info["category"]:
            folder = cat_info["folder_path"]
            with open(f"weights/{folder}/model_info.json") as f:
                model_info = json.load(f)
            for cname, minfo in model_info.items():
                if cname == character or minfo.get("title") == char_info["title"]:
                    model_path = f"weights/{folder}/{cname}/{minfo['model_path']}"
                    index_path = f"weights/{folder}/{cname}/{minfo['feature_retrieval_library']}"
                    break

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

    cb(40, "加载音频...")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    times = [0, 0, 0]

    cb(50, "正在转换...")
    audio_opt = vc.pipeline(
        hubert, net_g, 0, audio, audio_path,
        times, f0_up_key, f0_method, index_path, index_rate,
        if_f0, filter_radius, tgt_sr, resample_sr,
        rms_mix_rate, version, protect, f0_file=None,
    )
    cb(90, "保存输出...")

    output_path = f"temp/output_{uuid.uuid4().hex[:8]}.wav"
    sf.write(output_path, audio_opt, tgt_sr)

    from datetime import datetime
    info = (f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"npy: {times[0]:.2f}s | f0: {times[1]:.2f}s | infer: {times[2]:.2f}s")

    cb(100, "完成")
    return {"output_path": output_path, "sample_rate": tgt_sr, "info": info}


def task_uvr5_separate(params: dict, cb) -> dict:
    # Try remote first
    try:
        from backend.uvr5_remote import load_config, separate_remote
        cfg = load_config()
        if cfg.get("enabled") and cfg.get("api_url"):
            cb(10, "远程处理中...")
            import asyncio, uuid as uu
            audio_path = params["audio_path"]
            with open(audio_path, "rb") as af:
                audio_data = af.read()
            result = asyncio.run(separate_remote(audio_data, params.get("model_name", "mel_band_roformer")))
            def save_stem(blob, stem):
                p = f"temp/{stem}_{uu.uuid4().hex[:8]}.wav"
                with open(p, "wb") as f:
                    f.write(blob)
                return p
            vocal_path = save_stem(result.get("vocals", b""), "vocals")
            inst_path = save_stem(result.get("instrumental", b""), "inst")
            cb(100, "远程处理完成")
            return {"vocals": vocal_path, "instrumental": inst_path, "status": result.get("status", "远程分离完成")}
    except Exception as e:
        logger.warning(f"远程 UVR5 不可用，回退到本地: {e}")

    from uvr5 import separate_audio_vocals

    cb(5, "加载分离模型...")
    audio_path = params["audio_path"]
    model_name = params.get("model_name", "mel_band_roformer")
    dur = params.get("_duration_sec", 5)

    cb(20, "正在分离 (0%)...")
    vocals, inst, status = separate_audio_vocals(audio_path, model_name)
    if vocals is None:
        raise ValueError(status)

    cb(80, "保存结果...")
    def _save(tup, stem):
        import soundfile as sf
        sr, arr = tup
        p = f"temp/{stem}_{uuid.uuid4().hex[:8]}.wav"
        if arr.ndim == 2 and arr.shape[0] <= 2:
            arr = arr.T
        sf.write(p, arr, sr)
        return p

    vocal_path = _save(vocals, "vocals")
    inst_path = _save(inst, "inst")

    cb(100, "完成")
    return {"vocals": vocal_path, "instrumental": inst_path, "status": status}


def task_uvr5_dereverb(params: dict, cb) -> dict:
    # Try remote first
    try:
        from backend.uvr5_remote import load_config, dereverb_remote
        cfg = load_config()
        if cfg.get("enabled") and cfg.get("api_url"):
            cb(10, "远程处理中...")
            import asyncio, uuid as uu
            audio_path = params["audio_path"]
            with open(audio_path, "rb") as af:
                audio_data = af.read()
            result = asyncio.run(dereverb_remote(audio_data, params.get("overlap", 4)))
            def save_stem(blob, stem):
                p = f"temp/{stem}_{uu.uuid4().hex[:8]}.wav"
                with open(p, "wb") as f:
                    f.write(blob)
                return p
            dry_path = save_stem(result.get("dry", b""), "dry")
            reverb_path = save_stem(result.get("reverb", b""), "reverb")
            cb(100, "远程处理完成")
            return {"dry": dry_path, "reverb": reverb_path, "status": result.get("status", "远程去混响完成")}
    except Exception as e:
        logger.warning(f"远程 UVR5 去混响不可用，回退到本地: {e}")

    from uvr5 import separate_dereverb

    cb(5, "加载去混响模型...")
    audio_path = params["audio_path"]
    overlap = params.get("overlap", 4)

    cb(20, "正在消除混响...")
    dry, reverb, status = separate_dereverb(audio_path, overlap=overlap)
    if dry is None:
        raise ValueError(status)

    cb(80, "保存结果...")
    def _save(tup, stem):
        import soundfile as sf
        sr, arr = tup
        p = f"temp/{stem}_{uuid.uuid4().hex[:8]}.wav"
        if arr.ndim == 2 and arr.shape[0] <= 2:
            arr = arr.T
        sf.write(p, arr, sr)
        return p

    dry_path = _save(dry, "dry")
    reverb_path = _save(reverb, "reverb")

    cb(100, "完成")
    return {"dry": dry_path, "reverb": reverb_path, "status": status}


# ════════════════════════════════════════════
# 辅助
# ════════════════════════════════════════════

def check_ml_deps():
    global ml_available
    for mod in ["torch", "librosa", "numpy", "fairseq", "pyworld", "parselmouth"]:
        try:
            __import__(mod)
        except ImportError:
            ml_available = False
            return False
    ml_available = True
    return True


def get_characters_metadata():
    chars = []
    if not os.path.isfile("weights/folder_info.json"):
        return chars
    with open("weights/folder_info.json", encoding="utf-8") as f:
        folder_info = json.load(f)
    for cat_name, cat_info in folder_info.items():
        if not cat_info.get("enable", True):
            continue
        ct, cf, desc = cat_info["title"], cat_info["folder_path"], cat_info.get("description", "")
        mi_path = f"weights/{cf}/model_info.json"
        if not os.path.isfile(mi_path):
            continue
        with open(mi_path, encoding="utf-8") as f:
            models_info = json.load(f)
        cl = []
        for cn, info in models_info.items():
            if not info.get("enable", True):
                continue
            cv = f"weights/{cf}/{cn}/{info['cover']}"
            cl.append({"name": cn, "title": info["title"], "author": info.get("author", ""),
                       "cover": cv if os.path.isfile(cv) else None, "version": "v2", "category": ct})
        chars.append({"title": ct, "folder": cf, "description": desc, "characters": cl})
    return chars


# ════════════════════════════════════════════
# FastAPI
# ════════════════════════════════════════════

def _refresh_metadata():
    global categories_meta
    categories_meta = get_characters_metadata()


def _compress_image(img_bytes: bytes, max_size: int = 300) -> bytes:
    """Compress image to JPEG with max dimension and quality 80"""
    import io
    try:
        im = Image.open(io.BytesIO(img_bytes))
        im = im.convert("RGB")
        w, h = im.size
        if w > max_size or h > max_size:
            ratio = max_size / max(w, h)
            im = im.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80, optimize=True)
        return buf.getvalue()
    except Exception:
        return img_bytes

@asynccontextmanager
async def lifespan(app: FastAPI):
    global categories_meta
    categories_meta = get_characters_metadata()
    logger.info(f"已加载 {sum(len(c['characters']) for c in categories_meta)} 个角色")
    check_ml_deps()
    yield


app = FastAPI(title="Blue Archive RVC", version="2.0.0", lifespan=lifespan)
static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
weights_dir = Path(__file__).parent.parent / "weights"
if weights_dir.exists():
    app.mount("/weights", StaticFiles(directory=str(weights_dir)), name="weights")
os.makedirs("temp", exist_ok=True)


@app.get("/")
async def root():
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "ml_available": ml_available}

@app.get("/api/models")
async def list_models(search: str = ""):
    cats = get_characters_metadata()
    if search:
        q = search.lower()
        filtered = []
        for cat in cats:
            chars = [ch for ch in cat.get("characters", [])
                     if q in ch.get("name", "").lower()
                     or q in ch.get("title", "").lower()
                     or q in ch.get("author", "").lower()]
            if chars:
                nc = dict(cat)
                nc["characters"] = chars
                filtered.append(nc)
        return {"categories": filtered}
    return {"categories": cats}


@app.post("/api/upload")
async def upload_audio(request: Request, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1] or ".wav"
    dest = f"temp/upload_{os.urandom(4).hex()}{ext}"
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"path": dest, "filename": file.filename}



# ─── 任务提交 ───

def _estimate_dur(path):
    try:
        import librosa
        return librosa.get_duration(path=path)
    except:
        return 5.0


@app.post("/api/convert")
async def convert_voice(
    request: Request, character: str = Form(...), audio_path: str = Form(...),
    f0_up_key: int = Form(0), f0_method: str = Form("rmvpe"),
    index_rate: float = Form(0.7), filter_radius: int = Form(3),
    resample_sr: int = Form(0), rms_mix_rate: float = Form(1.0),
    protect: float = Form(0.5),
):
    if not ml_available:
        raise HTTPException(status_code=503, detail="ML 不可用")
    if not os.path.isfile(audio_path):
        raise HTTPException(status_code=400, detail="音频文件未找到")
    dur = _estimate_dur(audio_path)
    params = {"character": character, "audio_path": audio_path, "f0_up_key": f0_up_key,
              "f0_method": f0_method, "index_rate": index_rate, "filter_radius": filter_radius,
              "resample_sr": resample_sr, "rms_mix_rate": rms_mix_rate, "protect": protect,
              "_duration_sec": dur}
    qid = queue.enqueue("convert", params, task_convert)
    return {"queue_id": qid, "duration_sec": dur}


@app.post("/api/uvr5/separate")
async def uvr5_separate(audio_path: str = Form(...), model_name: str = Form("mel_band_roformer")):
    if not ml_available:
        raise HTTPException(status_code=503, detail="ML 不可用")
    if not os.path.isfile(audio_path):
        raise HTTPException(status_code=400, detail="音频文件未找到")
    dur = _estimate_dur(audio_path)
    params = {"audio_path": audio_path, "model_name": model_name, "_duration_sec": dur}
    qid = queue.enqueue("uvr5_separate", params, task_uvr5_separate)
    return {"queue_id": qid, "duration_sec": dur}


@app.post("/api/uvr5/dereverb")
async def uvr5_dereverb(request: Request, audio_path: str = Form(...), overlap: int = Form(4)):
    if not ml_available:
        raise HTTPException(status_code=503, detail="ML 不可用")
    if not os.path.isfile(audio_path):
        raise HTTPException(status_code=400, detail="音频文件未找到")
    dur = _estimate_dur(audio_path)
    params = {"audio_path": audio_path, "overlap": overlap, "_duration_sec": dur}
    qid = queue.enqueue("uvr5_dereverb", params, task_uvr5_dereverb)
    return {"queue_id": qid, "duration_sec": dur}


@app.get("/api/queue/{queue_id}")
async def queue_status(request: Request, queue_id: str):
    return queue.get_status(queue_id)


@app.get("/api/download/{filename:path}")
async def download(request: Request, filename: str):
    fp = f"temp/{filename}"
    if not os.path.isfile(fp):
        raise HTTPException(status_code=404, detail="文件未找到")
    return FileResponse(fp, media_type="audio/wav", filename=filename)



# ════════════════════════════════════════════

@app.post("/api/admin/login")
async def admin_login(password: str = Form(...)):
    if password == ADMIN_PASSWORD:
        return {"ok": True}
    raise HTTPException(status_code=401, detail="密码错误")


@app.get("/admin/login")
async def admin_login_page():
    return FileResponse(static_dir / "admin_login.html")


def _check_admin(request):
    """Verify admin password from query param or header"""
    pwd = request.query_params.get("pwd", "")
    if pwd == ADMIN_PASSWORD:
        return True
    auth = request.headers.get("X-Admin-Password", "")
    if auth == ADMIN_PASSWORD:
        return True
    return False


# ════════════════════════════════════════════
# GSV (GPT-SoVits) TTS
# ════════════════════════════════════════════

@app.get("/api/gsv/check")
async def gsv_check(request: Request):
    from backend.gsv import check_connection
    api_url = request.query_params.get("api_url", None)
    return await check_connection(api_url)


@app.get("/api/gsv/config")
async def gsv_config():
    from backend.gsv import load_config
    return load_config()


@app.get("/api/gsv/models")
async def gsv_models(search: str = "", section: str = ""):
    from backend.gsv import get_models
    ms = get_models()
    if section:
        ms = [m for m in ms if m.get("section", "") == section]
    if search:
        q = search.lower()
        ms = [m for m in ms if q in m.get("name", "").lower()]
    return {"models": ms, "available": len(ms) > 0}


@app.post("/api/gsv/tts")
async def gsv_tts(
    request: Request,
    text: str = Form(...),
    model_name: str = Form(...),
    speed_factor: float = Form(None),
    temperature: float = Form(None),
    top_k: int = Form(None),
):
    if len(text) > 500:
        raise HTTPException(status_code=400, detail=f"文本过长 ({len(text)} 字符)，最大支持 500 字符")
    from backend.gsv import load_config, switch_model, set_refer_audio, tts as gsv_tts_inner
    cfg = load_config()
    model_config = None
    for m in cfg.get("models", []):
        if m["name"] == model_name:
            model_config = dict(m)  # copy
            break
    if not model_config:
        raise HTTPException(status_code=404, detail="model not found")

    if speed_factor is not None:
        model_config["speed_factor"] = speed_factor
    if temperature is not None:
        model_config["temperature"] = temperature
    if top_k is not None:
        model_config["top_k"] = top_k

    sr = await switch_model(model_config["gpt_path"], model_config["sovits_path"])
    if not sr["ok"]:
        raise HTTPException(status_code=502, detail=sr["message"])
    rr = await set_refer_audio(model_config.get("ref_audio_path", ""))
    if not rr["ok"]:
        raise HTTPException(status_code=502, detail=rr["message"])
    try:
        from fastapi.responses import Response
        audio_bytes = await gsv_tts_inner(text, model_config)
        return Response(content=audio_bytes, media_type="audio/wav")
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


# Admin: GSV
@app.get("/api/admin/gsv/config")
async def admin_gsv_config(request: Request):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.gsv import load_config
    return load_config()


@app.put("/api/admin/gsv/config")
async def admin_gsv_update_config(request: Request, api_url: str = Form(...), timeout: int = Form(60)):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.gsv import load_config, save_config
    cfg = load_config()
    cfg["api_url"] = api_url
    cfg["timeout"] = timeout
    save_config(cfg)
    return {"ok": True}


@app.post("/api/admin/gsv/models")
async def admin_gsv_add_model(
    request: Request,
    name: str = Form(...),
    gpt_path: str = Form(...),
    sovits_path: str = Form(...),
    ref_audio_path: str = Form(""),
    prompt_text: str = Form(""),
    prompt_lang: str = Form("zh"),
    text_lang: str = Form("zh"),
    cover: str = Form(""),
    enable: bool = Form(True),
    cover_file: UploadFile = File(None),
):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Handle cover file upload
    cover_path = cover
    if cover_file and cover_file.filename:
        cover_dir = Path(__file__).parent.parent / "temp" / "gsv_covers"
        cover_dir.mkdir(parents=True, exist_ok=True)
        ext = os.path.splitext(cover_file.filename)[1] or ".jpg"
        cover_name = f"{name}{ext}"
        dest = cover_dir / cover_name
        with open(dest, "wb") as f:
            f.write(await cover_file.read())
        cover_path = str(dest)  # Store full temp path, frontend can serve via /api/download

    from backend.gsv import load_config, save_config
    cfg = load_config()
    for m in cfg["models"]:
        if m["name"] == name:
            raise HTTPException(status_code=400, detail="model exists")
    cfg["models"].append({
        "name": name, "gpt_path": gpt_path, "sovits_path": sovits_path,
        "ref_audio_path": ref_audio_path, "prompt_text": prompt_text,
        "prompt_lang": prompt_lang, "text_lang": text_lang,
        "cover": cover_path, "enable": enable,
    })
    save_config(cfg)
    return {"ok": True}


@app.put("/api/admin/gsv/models/{name}")
async def admin_gsv_update_model(
    request: Request,
    name: str,
    gpt_path: str = Form(None),
    sovits_path: str = Form(None),
    ref_audio_path: str = Form(None),
    prompt_text: str = Form(None),
    prompt_lang: str = Form(None),
    text_lang: str = Form(None),
    cover: str = Form(None),
    enable: str = Form(None),
    section: str = Form(None),
    cover_file: UploadFile = File(None),
):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.gsv import load_config, save_config
    cfg = load_config()
    for m in cfg["models"]:
        if m["name"] == name:
            if gpt_path is not None: m["gpt_path"] = gpt_path
            if sovits_path is not None: m["sovits_path"] = sovits_path
            if ref_audio_path is not None: m["ref_audio_path"] = ref_audio_path
            if prompt_text is not None: m["prompt_text"] = prompt_text
            if prompt_lang is not None: m["prompt_lang"] = prompt_lang
            if text_lang is not None: m["text_lang"] = text_lang
            if enable is not None:
                m["enable"] = enable.lower() in ("true", "1", "yes")
            if section is not None: m["section"] = section
            if cover is not None: m["cover"] = cover
            # Handle cover file upload with compression
            if cover_file and cover_file.filename:
                cover_dir = Path(__file__).parent.parent / "temp" / "gsv_covers"
                cover_dir.mkdir(parents=True, exist_ok=True)
                cover_name = f"{name}.jpg"
                dest = cover_dir / cover_name
                raw = await cover_file.read()
                compressed = _compress_image(raw)
                with open(dest, "wb") as f:
                    f.write(compressed)
                m["cover"] = str(dest)
            save_config(cfg)
            return {"ok": True}
    raise HTTPException(status_code=404, detail="not found")


@app.delete("/api/admin/gsv/models/{name}")
async def admin_gsv_delete_model(request: Request, name: str):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.gsv import load_config, save_config
    cfg = load_config()
    cfg["models"] = [m for m in cfg["models"] if m["name"] != name]
    save_config(cfg)
    return {"ok": True}



# ════════════════════════════════════════
# RVC Remote Config
# ════════════════════════════════════════

@app.get("/api/admin/rvc_remote/config")
async def admin_rvc_remote_config(request: Request):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.rvc_remote import load_config
    return load_config()


@app.put("/api/admin/rvc_remote/config")
async def admin_rvc_remote_update(
    request: Request,
    api_url: str = Form(...),
    api_key: str = Form(""),
    enabled: str = Form("false"),
    timeout: int = Form(120),
):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.rvc_remote import load_config, save_config
    cfg = load_config()
    cfg["api_url"] = api_url
    cfg["api_key"] = api_key
    cfg["enabled"] = enabled.lower() in ("true", "1", "yes")
    cfg["timeout"] = timeout
    save_config(cfg)
    return {"ok": True}


@app.get("/api/admin/rvc_remote/check")
async def admin_rvc_remote_check(request: Request):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.rvc_remote import check_connection
    return await check_connection()


# Admin 后台
# ════════════════════════════════════════════

@app.get("/api/admin/check")
async def admin_check(request: Request):
    return {"ok": _check_admin(request)}


@app.get("/admin")
async def admin_page():
    return FileResponse(static_dir / "admin.html")


@app.get("/api/admin/categories")
async def admin_categories(request: Request):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.admin import list_categories
    return {"categories": list_categories()}


@app.post("/api/admin/categories")
async def admin_create_category(request: Request, key: str = Form(...), title: str = Form(...),
                                folder_path: str = Form(None), description: str = Form("")):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.admin import create_category
    try:
        result = create_category(key, title, folder_path, description)
        _refresh_metadata()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/admin/categories/{key}/toggle")
async def admin_toggle_category(request: Request, key: str, enable: bool = Form(...)):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.admin import toggle_category
    try:
        result = toggle_category(key, enable)
        _refresh_metadata()
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/admin/categories/{key}")
async def admin_delete_category(request: Request, key: str):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.admin import delete_category
    try:
        result = delete_category(key)
        _refresh_metadata()
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/admin/categories/{key}/models")
async def admin_models(request: Request, key: str):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.admin import list_models
    try:
        return {"models": list_models(key)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.put("/api/admin/categories/{key}/models/{name}/toggle")
async def admin_toggle_model(request: Request, key: str, name: str, enable: bool = Form(...)):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.admin import toggle_model
    try:
        result = toggle_model(key, name, enable)
        _refresh_metadata()
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/admin/categories/{key}/models")
async def admin_add_model(request: Request, 
    key: str,
    char_name: str = Form(...),
    title: str = Form(...),
    author: str = Form(""),
    model_file: UploadFile = File(None),
    index_file: UploadFile = File(None),
    cover_file: UploadFile = File(None),
):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.admin import _load_folder_info, _get_model_dir, add_model

    fi = _load_folder_info()
    if key not in fi:
        raise HTTPException(status_code=404, detail=f"分类 '{key}' 不存在")
    folder = fi[key]["folder_path"]
    model_dir = _get_model_dir(folder, char_name)
    os.makedirs(model_dir, exist_ok=True)

    model_path = index_path = cover_path = ""

    if model_file:
        ext = os.path.splitext(model_file.filename)[1]
        dest = model_dir / f"model{ext}"
        with open(dest, "wb") as f:
            f.write(await model_file.read())
        model_path = str(dest.name)

    if index_file:
        dest = model_dir / index_file.filename
        with open(dest, "wb") as f:
            f.write(await index_file.read())
        index_path = str(dest.name)

    if cover_file:
        raw = await cover_file.read()
        compressed = _compress_image(raw)
        dest = model_dir / "cover.jpg"
        with open(dest, "wb") as f:
            f.write(compressed)
        cover_path = "cover.jpg"

    try:
        result = add_model(key, char_name, title, author,
                          model_path or "model.pth", index_path or "", cover_path or "")
        _refresh_metadata()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/admin/categories/{key}/models/{name}")
async def admin_update_model(
    request: Request, key: str, name: str,
    char_name: str = Form(None),
    title: str = Form(None),
    author: str = Form(None),
    model_file: UploadFile = File(None),
    index_file: UploadFile = File(None),
    cover_file: UploadFile = File(None),
):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.admin import _load_folder_info, _get_model_dir, update_model

    fi = _load_folder_info()
    if key not in fi:
        raise HTTPException(status_code=404, detail=f"分类 '{key}' 不存在")
    folder = fi[key]["folder_path"]
    new_name = char_name  # may be None (no rename) or a new name

    model_file_name = index_file_name = cover_file_name = None

    if model_file:
        model_dir = _get_model_dir(folder, new_name or name)
        model_dir.mkdir(parents=True, exist_ok=True)
        ext = os.path.splitext(model_file.filename)[1]
        dest = model_dir / f"model{ext}"
        with open(dest, "wb") as f:
            f.write(await model_file.read())
        model_file_name = str(dest.name)

    if index_file:
        model_dir = _get_model_dir(folder, new_name or name)
        model_dir.mkdir(parents=True, exist_ok=True)
        dest = model_dir / index_file.filename
        with open(dest, "wb") as f:
            f.write(await index_file.read())
        index_file_name = str(dest.name)

    if cover_file:
        model_dir = _get_model_dir(folder, new_name or name)
        model_dir.mkdir(parents=True, exist_ok=True)
        raw = await cover_file.read()
        compressed = _compress_image(raw)
        dest = model_dir / "cover.jpg"
        with open(dest, "wb") as f:
            f.write(compressed)
        cover_file_name = "cover.jpg"

    try:
        result = update_model(key, name, new_name=new_name,
                             title=title, author=author,
                             model_file_name=model_file_name,
                             index_file_name=index_file_name,
                             cover_file_name=cover_file_name)
        _refresh_metadata()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/admin/categories/{key}/models/{name}")
async def admin_delete_model(request: Request, key: str, name: str):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.admin import delete_model
    try:
        result = delete_model(key, name)
        _refresh_metadata()
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ════════════════════════════════════════
# UVR5 Remote Config
# ════════════════════════════════════════

@app.get("/api/admin/uvr5_remote/config")
async def admin_uvr5_remote_config(request: Request):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.uvr5_remote import load_config
    return load_config()


@app.put("/api/admin/uvr5_remote/config")
async def admin_uvr5_remote_update(
    request: Request,
    api_url: str = Form(...),
    enabled: bool = Form(False),
    timeout: int = Form(120),
):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.uvr5_remote import save_config
    save_config({"api_url": api_url, "enabled": enabled, "timeout": timeout})
    return {"ok": True}


@app.get("/api/admin/uvr5_remote/check")
async def admin_uvr5_remote_check(request: Request):
    if not _check_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from backend.uvr5_remote import check_connection
    return await check_connection()


def run():
    uvicorn.run("backend.api:app", host="0.0.0.0", port=7860, reload=False)
