"""
GSV (GPT-SoVits) 远程 TTS 集成模块
"""
import os, json, logging, httpx, urllib.parse
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "weights" / "gsv_config.json"

DEFAULT_CONFIG = {
    "api_url": "http://frp-add.com:54565",
    "timeout": 60,
    "models": []
}

# ─── 配置管理 ───

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            if "api_url" not in cfg:
                cfg["api_url"] = DEFAULT_CONFIG["api_url"]
            if "models" not in cfg:
                cfg["models"] = []
            if "timeout" not in cfg:
                cfg["timeout"] = DEFAULT_CONFIG["timeout"]
            return cfg
    return dict(DEFAULT_CONFIG)

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def get_models():
    cfg = load_config()
    return [m for m in cfg.get("models", []) if m.get("enable", True) and m.get("gpt_path") and m.get("sovits_path")]

# ─── API 通信 ───

async def check_connection(api_url: str = None) -> dict:
    """检查 GSV 服务是否可达"""
    if not api_url:
        api_url = load_config()["api_url"]
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{api_url}/openapi.json")
            if resp.status_code == 200:
                return {"ok": True, "message": "GSV 服务在线"}
            return {"ok": False, "message": f"返回状态码 {resp.status_code}"}
    except httpx.ConnectError:
        return {"ok": False, "message": "连接失败，服务器未响应"}
    except httpx.TimeoutException:
        return {"ok": False, "message": "连接超时"}
    except Exception as e:
        return {"ok": False, "message": str(e)}

async def switch_model(gpt_path: str, sovits_path: str, api_url: str = None) -> dict:
    """切换 GSV 的 GPT 和 SoVits 模型"""
    if not api_url:
        api_url = load_config()["api_url"]
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # 切换 GPT
            r1 = await client.get(f"{api_url}/set_gpt_weights", params={"weights_path": gpt_path})
            if r1.status_code != 200:
                return {"ok": False, "message": f"GPT 模型切换失败: {r1.text}"}
            # 切换 SoVits
            r2 = await client.get(f"{api_url}/set_sovits_weights", params={"weights_path": sovits_path})
            if r2.status_code != 200:
                return {"ok": False, "message": f"SoVits 模型切换失败: {r2.text}"}
            return {"ok": True, "message": "模型切换成功"}
    except Exception as e:
        return {"ok": False, "message": str(e)}

async def set_refer_audio(ref_audio_path: str, api_url: str = None) -> dict:
    """设置参考音频"""
    if not api_url:
        api_url = load_config()["api_url"]
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{api_url}/set_refer_audio", params={"refer_audio_path": ref_audio_path})
            if r.status_code == 200:
                return {"ok": True, "message": "参考音频设置成功"}
            return {"ok": False, "message": f"设置失败: {r.text}"}
    except Exception as e:
        return {"ok": False, "message": str(e)}

async def tts(text: str, model_config: dict, api_url: str = None, timeout: int = None) -> bytes:
    """
    调用 GSV TTS 接口
    返回 wav 音频字节
    """
    cfg = load_config()
    if not api_url:
        api_url = cfg["api_url"]
    if timeout is None:
        timeout = cfg.get("timeout", 60)

    params = {
        "text": text,
        "text_lang": model_config.get("text_lang", "zh"),
        "ref_audio_path": model_config.get("ref_audio_path", ""),
        "prompt_text": model_config.get("prompt_text", ""),
        "prompt_lang": model_config.get("prompt_lang", "zh"),
        "top_k": model_config.get("top_k", 5),
        "top_p": model_config.get("top_p", 1),
        "temperature": model_config.get("temperature", 1),
        "text_split_method": "cut5",
        "batch_size": 1,
        "media_type": "wav",
        "speed_factor": model_config.get("speed_factor", 1.0),
        "seed": -1,
        "parallel_infer": True,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(f"{api_url}/tts", params=params)
        if resp.status_code != 200:
            detail = resp.text
            try:
                detail = resp.json().get("message", detail)
            except:
                pass
            raise RuntimeError(f"GSV TTS 失败: {detail}")
        return resp.content
