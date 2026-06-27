"""
UVR5 远程 API 模块 — 人声分离 / 混响消除
先尝试远程 UVR5 API 服务器，失败则回退到本地
"""
import os
import json
import logging
import httpx
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "weights" / "uvr5_remote_config.json"
DEFAULT_CONFIG = {
    "api_url": "",
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

async def check_connection(api_url: str = None) -> dict:
    """检查远程 UVR5 服务是否可达"""
    cfg = load_config()
    api_url = api_url or cfg["api_url"]
    if not api_url:
        return {"ok": False, "message": "未配置远程 API 地址"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{api_url.rstrip('/')}/api/uvr5/health")
            if resp.status_code == 200:
                data = resp.json()
                return {"ok": True, "message": f"远程 UVR5 服务在线 (models={'可用' if data.get('models_loaded') else '不可用'})"}
            return {"ok": False, "message": f"状态码 {resp.status_code}"}
    except httpx.ConnectError:
        return {"ok": False, "message": "连接失败，服务器未响应"}
    except httpx.TimeoutException:
        return {"ok": False, "message": "连接超时"}
    except Exception as e:
        return {"ok": False, "message": str(e)}


async def _download_file(client: httpx.AsyncClient, api_url: str, file_path: str) -> bytes:
    """从远程服务器下载结果文件"""
    if not file_path:
        return b""
    filename = os.path.basename(file_path)
    resp = await client.get(f"{api_url.rstrip('/')}/api/download/{filename}")
    if resp.status_code == 200:
        return resp.content
    logger.warning(f"下载远程文件失败: {filename} ({resp.status_code})")
    return b""


async def separate_remote(audio_data: bytes, model_name: str = "mel_band_roformer") -> dict:
    """
    调用远程 UVR5 人声分离
    返回: {"vocals": bytes, "instrumental": bytes, "status": str}
    """
    cfg = load_config()
    api_url = cfg["api_url"]
    timeout = cfg.get("timeout", 120)

    async with httpx.AsyncClient(timeout=timeout) as client:
        # 上传音频文件
        files = {"audio": ("input.wav", audio_data, "audio/wav")}
        resp = await client.post(
            f"{api_url.rstrip('/')}/api/uvr5/separate",
            data={"model_name": model_name},
            files=files,
        )
        if resp.status_code != 200:
            detail = resp.text
            try:
                detail = resp.json().get("detail", detail)
            except Exception:
                pass
            raise RuntimeError(f"远程 UVR5 分离失败: {detail}")

        result = resp.json()

        # 下载结果文件
        vocals_bytes = await _download_file(client, api_url, result.get("vocals", ""))
        inst_bytes = await _download_file(client, api_url, result.get("instrumental", ""))

        return {
            "vocals": vocals_bytes,
            "instrumental": inst_bytes,
            "status": result.get("status", "远程分离完成"),
        }


async def dereverb_remote(audio_data: bytes, overlap: int = 4) -> dict:
    """
    调用远程 UVR5 混响消除
    返回: {"dry": bytes, "reverb": bytes, "status": str}
    """
    cfg = load_config()
    api_url = cfg["api_url"]
    timeout = cfg.get("timeout", 120)

    async with httpx.AsyncClient(timeout=timeout) as client:
        files = {"audio": ("input.wav", audio_data, "audio/wav")}
        resp = await client.post(
            f"{api_url.rstrip('/')}/api/uvr5/dereverb",
            data={"overlap": overlap},
            files=files,
        )
        if resp.status_code != 200:
            detail = resp.text
            try:
                detail = resp.json().get("detail", detail)
            except Exception:
                pass
            raise RuntimeError(f"远程 UVR5 去混响失败: {detail}")

        result = resp.json()

        dry_bytes = await _download_file(client, api_url, result.get("dry", ""))
        reverb_bytes = await _download_file(client, api_url, result.get("reverb", ""))

        return {
            "dry": dry_bytes,
            "reverb": reverb_bytes,
            "status": result.get("status", "远程去混响完成"),
        }
