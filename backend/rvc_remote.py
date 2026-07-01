"""
Remote RVC API — 远程调用 & 本地回退
"""
import os, json, logging, httpx
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "weights" / "rvc_remote_config.json"
DEFAULT_CONFIG = {
    "api_url": "",
    "api_key": "",
    "enabled": False,
    "timeout": 120,
}

# ─── 配置管理 ───

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            for k in DEFAULT_CONFIG:
                cfg.setdefault(k, DEFAULT_CONFIG[k])
            return cfg
    return dict(DEFAULT_CONFIG)

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

# ─── 远程调用 ───

async def check_connection(api_url: str = None, api_key: str = None) -> dict:
    """检查远程 RVC 服务是否可达"""
    cfg = load_config()
    api_url = api_url or cfg["api_url"]
    if not api_url:
        return {"ok": False, "message": "未配置远程 API 地址"}
    try:
        headers = {}
        if api_key or cfg.get("api_key"):
            headers["X-Api-Key"] = api_key or cfg["api_key"]
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{api_url.rstrip('/')}/api/health", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                return {"ok": True, "message": f"远程 RVC 服务在线 (ML={'可用' if data.get('ml_available') else '不可用'})"}
            return {"ok": False, "message": f"状态码 {resp.status_code}"}
    except httpx.ConnectError:
        return {"ok": False, "message": "连接失败"}
    except httpx.TimeoutException:
        return {"ok": False, "message": "连接超时"}
    except Exception as e:
        return {"ok": False, "message": str(e)}

async def convert_remote(character: str, audio_data: bytes, params: dict) -> bytes:
    """远程调用变声转换，返回音频字节
    本地只传模型文件名，远程自己在 weights/ 下搜索
    """
    cfg = load_config()
    api_url = cfg["api_url"].rstrip("/")
    headers = {}
    if cfg.get("api_key"):
        headers["X-Api-Key"] = cfg["api_key"]

    # 本地查找模型文件名（不含路径）
    model_file = ""
    index_file = ""
    try:
        from backend.api import categories_meta
        for cat in categories_meta:
            for ch in cat.get("characters", []):
                if ch["name"] == character:
                    # 从 folder_info + model_info 查文件名
                    import json
                    with open("weights/folder_info.json") as f:
                        fi = json.load(f)
                    for ck, cv in fi.items():
                        if cv["title"] == ch["category"]:
                            folder = cv["folder_path"]
                            with open(f"weights/{folder}/model_info.json") as f2:
                                mi = json.load(f2)
                            if character in mi:
                                model_file = mi[character].get("model_path", "")
                                index_file = mi[character].get("feature_retrieval_library", "")
                            break
                    break
    except Exception as e:
        logger.warning(f"查找模型文件名失败: {e}")

    files = {"audio": ("input.wav", audio_data, "audio/wav")}
    form = {
        "character": character,
        "model_file": model_file,
        "index_file": index_file,
        "f0_up_key": params.get("f0_up_key", 0),
        "f0_method": params.get("f0_method", "rmvpe"),
        "index_rate": params.get("index_rate", 0.7),
        "filter_radius": params.get("filter_radius", 3),
        "resample_sr": params.get("resample_sr", 0),
        "rms_mix_rate": params.get("rms_mix_rate", 1.0),
        "protect": params.get("protect", 0.5),
    }

    timeout_val = cfg.get("timeout", 120)
    async with httpx.AsyncClient(timeout=timeout_val) as client:
        resp = await client.post(f"{api_url}/api/rvc/convert", data=form, files=files, headers=headers)
        if resp.status_code != 200:
            detail = resp.text
            try: detail = resp.json().get("detail", detail)
            except: pass
            raise RuntimeError(f"远程 RVC 失败: {detail}")
        return resp.content

async def uvr5_remote(audio_data: bytes, model_name: str = "mel_band_roformer") -> dict:
    """远程调用人声分离，返回 {vocals, instrumental} 音频字节"""
    cfg = load_config()
    api_url = cfg["api_url"].rstrip("/")
    headers = {}
    if cfg.get("api_key"):
        headers["X-Api-Key"] = cfg["api_key"]

    files = {"file": ("input.wav", audio_data, "audio/wav")}
    async with httpx.AsyncClient(timeout=cfg.get("timeout", 120)) as client:
        r1 = await client.post(f"{api_url}/api/upload", files=files, headers=headers)
        if r1.status_code != 200:
            raise RuntimeError(f"上传失败: {r1.text}")
        upload_data = r1.json()

        form = {"audio_path": upload_data["path"], "model_name": model_name}
        r2 = await client.post(f"{api_url}/api/uvr5/separate", data=form, headers=headers)
        if r2.status_code != 200:
            raise RuntimeError(f"分离失败: {r2.text}")
        sep_data = r2.json()
        qid = sep_data.get("queue_id")
        if not qid:
            raise RuntimeError("远程未返回 queue_id")

        import asyncio
        for _ in range(40):
            await asyncio.sleep(3)
            r3 = await client.get(f"{api_url}/api/queue/{qid}", headers=headers)
            qd = r3.json()
            if qd.get("status") == "done" or qd.get("vocals"):
                result = {}
                for stem in ["vocals", "instrumental"]:
                    if qd.get(stem):
                        fn = qd[stem].split("/")[-1]
                        r4 = await client.get(f"{api_url}/api/download/{fn}", headers=headers)
                        if r4.status_code == 200:
                            result[stem] = r4.content
                return result
            if qd.get("status") == "error":
                raise RuntimeError(qd.get("error", "远程处理失败"))
        raise RuntimeError("远程处理超时")
