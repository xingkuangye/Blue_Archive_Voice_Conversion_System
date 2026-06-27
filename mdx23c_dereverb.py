"""
MDX23C De-Reverb 模块
使用 TFC_TDF_net 架构的混响消除模型
"""
import os
import sys
import yaml
import logging
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as scipy_signal
from ml_collections import ConfigDict

logger = logging.getLogger(__name__)

# ─── 全局缓存 ───
_model = None
_config = None


def unload_model():
    """释放 GPU 显存"""
    global _model, _config
    if _model is not None:
        import gc, torch
        _model = _model.cpu()
        del _model
        del _config
        _model = None
        _config = None
        gc.collect()
        torch.cuda.empty_cache()
        import logging
        logging.getLogger(__name__).info("MDX23C DeReverb: 模型已卸载，显存已释放")
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_MODEL_NAME = "MDX23C-De-Reverb-aufr33-jarredou.ckpt"
_CONFIG_NAME = "config_dereverb_mdx23c.yaml"


def _sanitize_yaml(raw):
    """清理 YAML 中的 Python 特有标签"""
    raw = raw.replace("!!python/tuple", "")
    return raw


def load_model(device=None):
    """
    加载 MDX23C De-Reverb 模型
    """
    global _model, _config

    if _model is not None:
        return _model, _config

    if device is None:
        device = torch.device("cpu")

    model_path = os.path.join(_MODEL_DIR, _MODEL_NAME)
    config_path = os.path.join(os.path.dirname(__file__), _CONFIG_NAME)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"MDX23C-DeReverb model not found at {model_path}. "
            f"Please download it from: "
            f"https://huggingface.co/Eddycrack864/audio-separator-models/resolve/main/mdx23c/MDX23C-De-Reverb-aufr33-jarredou.ckpt"
        )

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    # 加载配置
    with open(config_path, "r") as f:
        raw = f.read()
    config_dict = yaml.safe_load(_sanitize_yaml(raw))
    _config = ConfigDict(config_dict)

    # 加载模型
    from tfc_tdf_v3 import TFC_TDF_net

    model = TFC_TDF_net(_config, device=device)
    state = torch.load(model_path, map_location="cpu")

    if isinstance(state, dict):
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        elif "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
    else:
        model.load_state_dict(state)

    model.to(device).eval()
    logger.info(f"MDX23C DeReverb loaded: {sum(p.numel() for p in model.parameters()):,} params")

    _model = model
    return _model, _config


def process(audio_path, segment_size=None, overlap=4, pitch_shift=0, device=None):
    """
    对音频执行去混响处理

    Args:
        audio_path: 音频文件路径
        segment_size: 分段大小（帧数），None 则使用模型默认
        overlap: 重叠倍数
        pitch_shift: 音高偏移（半音）
        device: torch 设备

    Returns:
        (dry_audio, reverb_audio, sample_rate)
        dry_audio: ndarray, shape (channels, samples)
        reverb_audio: ndarray, shape (channels, samples)
        sample_rate: int
    """
    import librosa

    if device is None:
        device = torch.device("cpu")

    model, cfg = load_model(device)
    sr = cfg.audio.sample_rate  # 44100

    # 1. 加载音频
    audio, orig_sr = librosa.load(audio_path, sr=None, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.ndim == 2 and audio.shape[0] > 2:
        audio = audio.T
    if audio.shape[0] == 1:
        audio = np.concatenate([audio, audio], axis=0)

    # 重采样到模型采样率
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr, res_type="soxr_hq")

    # 归一化
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95

    # 2. 音高偏移
    if pitch_shift != 0:
        audio = _pitch_shift(audio, sr, -pitch_shift)

    mix = audio.copy()

    # 3. 确定分段参数
    if segment_size is None:
        segment_size = cfg.inference.dim_t  # 256
    if overlap is None:
        overlap = cfg.inference.get("num_overlap", 4)

    chunk_size = cfg.audio.hop_length * (segment_size - 1)  # 1024 * 255 = 261120
    hop_size = chunk_size // overlap

    # 4. 音频填充
    mix_t = torch.from_numpy(mix).float()
    mix_len = mix_t.shape[1]
    pad_size = hop_size - (mix_len - chunk_size) % hop_size
    mix_t = torch.cat([
        torch.zeros(2, chunk_size - hop_size),
        mix_t,
        torch.zeros(2, pad_size + chunk_size - hop_size),
    ], dim=1)

    chunks = mix_t.unfold(1, chunk_size, hop_size).transpose(0, 1)
    num_chunks = len(chunks)
    batch_size = cfg.inference.get("batch_size", 2)

    # 5. 推理
    num_stems = model.num_target_instruments  # 2: dry, No dry
    accumulated = torch.zeros(num_stems, *mix_t.shape)

    with torch.no_grad():
        for i in range(0, num_chunks, batch_size):
            batch = chunks[i:i+batch_size].to(device)
            output = model(batch)  # (batch, stems, channels, samples)
            output = output.cpu()

            for j, single_out in enumerate(output):
                idx = i + j
                start = idx * hop_size
                accumulated[..., start:start + chunk_size] += single_out

    # 重叠归一化
    inferenced = accumulated[..., chunk_size - hop_size: -(pad_size + chunk_size - hop_size)] / overlap

    # 6. 提取干声和混响
    instruments = cfg.training.instruments  # ['dry', 'No dry']
    dry_idx = instruments.index("dry") if "dry" in instruments else 0
    reverb_idx = instruments.index("No dry") if "No dry" in instruments else 1

    dry = inferenced[dry_idx].numpy()
    reverb = inferenced[reverb_idx].numpy()

    # 修正长度
    if dry.shape[1] > mix_len:
        dry = dry[:, :mix_len]
        reverb = reverb[:, :mix_len]
    elif dry.shape[1] < mix_len:
        pad_w = mix_len - dry.shape[1]
        dry = np.pad(dry, ((0,0), (0, pad_w)), mode="constant")
        reverb = np.pad(reverb, ((0,0), (0, pad_w)), mode="constant")

    # 音高还原
    if pitch_shift != 0:
        dry = _pitch_shift(dry, sr, pitch_shift)
        reverb = _pitch_shift(reverb, sr, pitch_shift)

    return dry, reverb, sr


def _pitch_shift(audio, sr, semitones):
    """简单音高偏移"""
    import librosa
    # librosa.pitch_shift 需要 (samples,) 或 (samples, channels)
    if audio.ndim == 2:
        shifted = np.array([librosa.effects.pitch_shift(audio[c], sr=sr, n_steps=semitones)
                           for c in range(audio.shape[0])])
    else:
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
    return shifted
