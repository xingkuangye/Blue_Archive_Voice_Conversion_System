"""
Admin 后台 — 模型管理
"""
import os, json, shutil, logging
from pathlib import Path

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
FOLDER_INFO = WEIGHTS_DIR / "folder_info.json"


def _load_folder_info():
    with open(FOLDER_INFO, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_folder_info(data):
    with open(FOLDER_INFO, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_model_info(category_folder):
    path = WEIGHTS_DIR / category_folder / "model_info.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_model_info(category_folder, data):
    path = WEIGHTS_DIR / category_folder / "model_info.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_model_dir(category_folder, char_name):
    return WEIGHTS_DIR / category_folder / char_name


def list_categories():
    """获取分类列表（含角色数）"""
    fi = _load_folder_info()
    result = []
    for key, info in fi.items():
        mi_path = WEIGHTS_DIR / info["folder_path"] / "model_info.json"
        char_count = 0
        if mi_path.exists():
            with open(mi_path, encoding="utf-8") as f:
                char_count = len(json.load(f))
        result.append({
            "key": key,
            "title": info["title"],
            "folder_path": info["folder_path"],
            "description": info.get("description", ""),
            "enable": info.get("enable", True),
            "char_count": char_count,
        })
    return result


def create_category(key: str, title: str, folder_path: str = None, description: str = ""):
    fi = _load_folder_info()
    if key in fi:
        raise ValueError(f"分类 '{key}' 已存在")
    if not folder_path:
        folder_path = key
    cat_dir = WEIGHTS_DIR / folder_path
    cat_dir.mkdir(parents=True, exist_ok=True)
    # 创建空的 model_info.json
    with open(cat_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)
    fi[key] = {"enable": True, "title": title, "folder_path": folder_path, "description": description}
    _save_folder_info(fi)
    logger.info(f"创建分类: {key} -> {folder_path}")
    return {"key": key, "title": title, "folder_path": folder_path}


def toggle_category(key: str, enable: bool):
    fi = _load_folder_info()
    if key not in fi:
        raise ValueError(f"分类 '{key}' 不存在")
    fi[key]["enable"] = enable
    _save_folder_info(fi)
    logger.info(f"分类 {key} -> enable={enable}")
    return {"key": key, "enable": enable}


def delete_category(key: str):
    fi = _load_folder_info()
    if key not in fi:
        raise ValueError(f"分类 '{key}' 不存在")
    folder_path = fi[key]["folder_path"]
    # 删除整个目录
    cat_dir = WEIGHTS_DIR / folder_path
    if cat_dir.exists():
        shutil.rmtree(cat_dir)
    del fi[key]
    _save_folder_info(fi)
    logger.info(f"删除分类: {key}")
    return {"key": key, "deleted": True}


def list_models(category_key: str):
    fi = _load_folder_info()
    if category_key not in fi:
        raise ValueError(f"分类 '{category_key}' 不存在")
    folder_path = fi[category_key]["folder_path"]
    mi = _load_model_info(folder_path)
    result = []
    for char_name, info in mi.items():
        model_dir = _get_model_dir(folder_path, char_name)
        model_size = 0
        index_size = 0
        if model_dir.exists():
            model_file = model_dir / info["model_path"]
            if model_file.exists():
                model_size = model_file.stat().st_size
            index_file = model_dir / info.get("feature_retrieval_library", "")
            if index_file.exists():
                index_size = index_file.stat().st_size
        result.append({
            "name": char_name,
            "title": info["title"],
            "author": info.get("author", ""),
            "enable": info.get("enable", True),
            "model_path": info["model_path"],
            "model_size": model_size,
            "index_size": index_size,
            "cover": info.get("cover", ""),
            "version": "v2",
        })
    return result


def toggle_model(category_key: str, char_name: str, enable: bool):
    fi = _load_folder_info()
    if category_key not in fi:
        raise ValueError(f"分类 '{category_key}' 不存在")
    folder_path = fi[category_key]["folder_path"]
    mi = _load_model_info(folder_path)
    if char_name not in mi:
        raise ValueError(f"模型 '{char_name}' 不存在")
    mi[char_name]["enable"] = enable
    _save_model_info(folder_path, mi)
    return {"category": category_key, "name": char_name, "enable": enable}


def delete_model(category_key: str, char_name: str):
    fi = _load_folder_info()
    if category_key not in fi:
        raise ValueError(f"分类 '{category_key}' 不存在")
    folder_path = fi[category_key]["folder_path"]
    mi = _load_model_info(folder_path)
    if char_name not in mi:
        raise ValueError(f"模型 '{char_name}' 不存在")
    # 删除模型目录
    model_dir = _get_model_dir(folder_path, char_name)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    del mi[char_name]
    _save_model_info(folder_path, mi)
    logger.info(f"删除模型: {char_name} from {category_key}")
    return {"category": category_key, "name": char_name, "deleted": True}


def add_model(category_key: str, char_name: str, title: str, author: str = "",
              model_file_path: str = "", index_file_path: str = "", cover_path: str = ""):
    fi = _load_folder_info()
    if category_key not in fi:
        raise ValueError(f"分类 '{category_key}' 不存在")
    folder_path = fi[category_key]["folder_path"]

    mi = _load_model_info(folder_path)
    if char_name in mi:
        raise ValueError(f"模型 '{char_name}' 已存在")

    mi[char_name] = {
        "enable": True,
        "model_path": os.path.basename(model_file_path) if model_file_path else "",
        "title": title,
        "cover": os.path.basename(cover_path) if cover_path else "",
        "feature_retrieval_library": os.path.basename(index_file_path) if index_file_path else "",
        "author": author,
    }
    _save_model_info(folder_path, mi)
    logger.info(f"添加模型: {char_name} to {category_key}")
    return {"category": category_key, "name": char_name}
