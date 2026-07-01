"""
Remote RVC API — 远程调用 & 本地回退
本地只传 model_file / index_file 文件名，远程自己在 weights/ 下搜索
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


def _find_model_files(character):
    """
    在本地的 weights/ 下查找角色对应的模型文件名
    返回 (model_file, index_file) 纯文件名，不含路径
    """
    try:
        fi_path = "weights/folder_info.json"
        if not os.path.isfile(fi_path):
            return "", ""
        with open(fi_path, encoding="utf-8") as f:
            folder_info = json.load(f)
        for cat_name, cat_info in folder_info.items():
            if not cat_info.get("enable", True):
                continue
            folder = cat_info["folder_path"]
            mi_path = f"weights/{folder}/model_info.json"
            if not os.path.isfile(mi_path):
                continue
            with open(mi_path, encoding="utf-8") as f:
                models_info = json.load(f)
            if character in models_info:
                info = models_info[character]
                return info.get("model_path", ""), info.get("feature_retrieval_library", "")
            # 也按 title 匹配
            for name, info in models_info.items():
                if info.get("title") == character or name == character:
                    return info.get("model_path", ""), info.get("feature_retrieval_library", "")
    except Exception as e:
        logger.warning(f"查找模型文件失败: {e}")
    return "", ""


async def check_connection(api_url: str = None, api_key: str = None) -> dict:
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
                return {"ok": True, "message": f"远程 RVC 服务在线 (cuda={data.get('cuda',False)})"}
            return {"ok": False, "message": f"状态码 {resp.status_code}"}
    except httpx.ConnectError:
        return {"ok": False, "message": "连接失败"}
    except httpx.TimeoutException:
        return {"ok": False, "message": "连接超时"}
    except Exception as e:
        return {"ok": False, "message": str(e)}


async def convert_remote(character: str, audio_data: bytes, params: dict) -> bytes:
    """
    远程调用变声转换
    本地只传 model_file / index_file 文件名，远程自己在 weights/ 下搜索
    """
    cfg = load_config()
    api_url = cfg["api_url"].rstrip("/")
    headers = {}
    if cfg.get("api_key"):
        headers["X-Api-Key"] = cfg["api_key"]

    # 本地查找模型文件名（不含路径）
    model_file, index_file = _find_model_files(character)

    if not model_file:
        raise RuntimeError(f"未找到角色 '{character}' 的模型文件")

    files = {"file": ("input.wav", audio_data, "audio/wav")}
    form = {
        "character": character,
        "audio_path": "",
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
        # 上传音频
        r1 = await client.post(f"{api_url}/api/upload", files=files, headers=headers)
        if r1.status_code != 200:
            raise RuntimeError(f"上传失败: {r1.text}")
        upload_data = r1.json()
        form["audio_path"] = upload_data["path"]

        # 提交转换
        r2 = await client.post(f"{api_url}/api/convert", data=form, headers=headers)
        if r2.status_code != 200:
            detail = r2.text
            try: detail = r2.json().get("detail", detail)
            except: pass
            raise RuntimeError(f"远程 RVC 失败: {detail}")
        conv_data = r2.json()
        qid = conv_data["queue_id"]

        # 轮询结果
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
