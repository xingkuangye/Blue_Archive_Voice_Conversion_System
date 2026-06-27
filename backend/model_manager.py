"""
模型管理器 — 加载和管理 RVC 模型
"""
import os
import json
import torch
import logging

from fairseq import checkpoint_utils
from fairseq.data.dictionary import Dictionary
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from config import Config
from vc_infer_pipeline import VC

logger = logging.getLogger(__name__)

# 全局变量
hubert_model = None
models_info = []  # [(category, folder, description, [(char_name, title, author, cover, version, vc_obj, tgt_sr, if_f0, index_path)])]


def load_hubert(config: Config):
    """加载 HuBERT 模型"""
    global hubert_model
    torch.serialization.add_safe_globals([Dictionary])
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"], suffix="",
    )
    hubert_model = models[0].to(config.device)
    hubert_model = hubert_model.half() if config.is_half else hubert_model.float()
    hubert_model.eval()
    logger.info("HuBERT model loaded successfully")
    return hubert_model


def load_models(config: Config):
    """加载所有 RVC 模型"""
    global models_info
    models_info = []
    categories = []

    if not os.path.isfile("weights/folder_info.json"):
        logger.warning("weights/folder_info.json not found")
        return categories

    with open("weights/folder_info.json", "r", encoding="utf-8") as f:
        folder_info = json.load(f)

    for category_name, category_info in folder_info.items():
        if not category_info.get("enable", True):
            continue
        category_title = category_info["title"]
        category_folder = category_info["folder_path"]
        description = category_info.get("description", "")

        info_path = f"weights/{category_folder}/model_info.json"
        if not os.path.isfile(info_path):
            continue

        with open(info_path, "r", encoding="utf-8") as f:
            models_info_data = json.load(f)

        chars = []
        for character_name, info in models_info_data.items():
            if not info.get("enable", True):
                continue
            try:
                model_title = info["title"]
                model_name = info["model_path"]
                model_author = info.get("author", "")
                model_cover = f"weights/{category_folder}/{character_name}/{info['cover']}"
                model_index = f"weights/{category_folder}/{character_name}/{info['feature_retrieval_library']}"
                model_path = f"weights/{category_folder}/{character_name}/{model_name}"

                cpt = torch.load(model_path, map_location="cpu")
                tgt_sr = cpt["config"][-1]
                cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
                if_f0 = cpt.get("f0", 1)
                version = cpt.get("version", "v1")

                if version == "v1":
                    net_g = (
                        SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
                        if if_f0 == 1
                        else SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
                    )
                else:
                    net_g = (
                        SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
                        if if_f0 == 1
                        else SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
                    )

                del net_g.enc_q
                net_g.load_state_dict(cpt["weight"], strict=False)
                net_g.eval().to(config.device)
                net_g = net_g.half() if config.is_half else net_g.float()

                vc = VC(tgt_sr, config)

                chars.append({
                    "name": character_name,
                    "title": model_title,
                    "author": model_author,
                    "cover": model_cover,
                    "version": version,
                    "category": category_title,
                    "category_folder": category_folder,
                    "if_f0": if_f0,
                    "tgt_sr": tgt_sr,
                    "index_path": model_index,
                    "net_g": net_g,
                    "vc": vc,
                })
                logger.info(f"Loaded: {character_name} ({version})")
            except Exception as e:
                logger.error(f"Failed to load {character_name}: {e}")
                continue

        categories.append({
            "title": category_title,
            "folder": category_folder,
            "description": description,
            "characters": chars,
        })

    models_info = categories
    return categories


def get_characters_metadata():
    """获取角色元数据（不含模型对象，用于 API 响应）"""
    result = []
    for cat in models_info:
        chars = []
        for ch in cat["characters"]:
            # 检查封面是否真实存在
            cover_path = ch["cover"]
            cover_exists = os.path.isfile(cover_path) if cover_path else False
            chars.append({
                "name": ch["name"],
                "title": ch["title"],
                "author": ch["author"],
                "cover": cover_path if cover_exists else None,
                "version": ch["version"],
                "category": ch["category"],
            })
        result.append({
            "title": cat["title"],
            "folder": cat["folder"],
            "description": cat["description"],
            "characters": chars,
        })
    return result


def find_character(char_name: str):
    """通过名称查找角色模型对象"""
    for cat in models_info:
        for ch in cat["characters"]:
            if ch["name"] == char_name:
                return ch
    return None
