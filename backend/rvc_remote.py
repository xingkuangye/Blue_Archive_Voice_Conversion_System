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
    """远程调用变声转换，返回音频字节"""
    cfg = load_config()
    api_url = cfg["api_url"].rstrip("/")
    headers = {}
    if cfg.get("api_key"):
        headers["X-Api-Key"] = cfg["api_key"]

    files = {"file": ("input.wav", audio_data, "audio/wav")}
    form = {
        "character": character,
        "f0_up_key": str(params.get("f0_up_key", 0)),
        "f0_method": params.get("f0_method", "pm"),
        "index_rate": str(params.get("index_rate", 0.7)),
        "filter_radius": str(params.get("filter_radius", 3)),
        "resample_sr": str(params.get("resample_sr", 0)),
        "rms_mix_rate": str(params.get("rms_mix_rate", 1.0)),
        "protect": str(params.get("protect", 0.5)),
    }

    timeout_val = cfg.get("timeout", DEFAULT_CONFIG["timeout"])
    async with httpx.AsyncClient(timeout=timeout_val) as client:
        r1 = await client.post(f"{api_url}/api/upload", files=files, headers=headers)
        if r1.status_code != 200:
            raise RuntimeError(f"上传失败: {r1.text}")
        upload_data = r1.json()
        audio_path = upload_data["path"]

        # Submit convert
        form["audio_path"] = audio_path
        r2 = await client.post(f"{api_url}/api/convert", data=form, headers=headers)
        if r2.status_code != 200:
            raise RuntimeError(f"转换失败: {r2.text}")
        conv_data = r2.json()
        qid = conv_data["queue_id"]

        # Poll until done
        import asyncio
        max_polls = (timeout_val + 5) // 3
        for _ in range(max_polls):
            await asyncio.sleep(3)
            r3 = await client.get(f"{api_url}/api/queue/{qid}", headers=headers)
            if r3.status_code != 200:
                raise RuntimeError(f"查询队列失败: {r3.text}")
            qd = r3.json()
            if qd.get("status") == "done" or qd.get("output_path"):
                out_fn = qd["output_path"].split("/")[-1]
                r4 = await client.get(f"{api_url}/api/download/{out_fn}", headers=headers)
                if r4.status_code == 200:
                    return r4.content
                raise RuntimeError(f"下载失败: {r4.status_code}")
            if qd.get("status") == "error":
                raise RuntimeError(qd.get("error", "远程处理失败"))

        raise RuntimeError("远程处理超时")

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
