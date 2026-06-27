"""
Mel-Band Roformer 人声分离模块
使用 vocals_mel_band_roformer.ckpt (Kimberley Jensen)
多核 CPU 并行批处理加速
"""
import os
os.environ.setdefault("TORCH_LOGS", "ERROR")
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")

import yaml
import logging
import numpy as np
import torch
from multiprocessing import cpu_count
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_MODEL_NAME = "vocals_mel_band_roformer.ckpt"
_CONFIG_NAME = "vocals_mel_band_roformer.yaml"

_model = None
_config = None

# ─── 全局 CPU 线程优化 ───
_NUM_CORES = cpu_count()
torch.set_num_threads(_NUM_CORES)
try:
    torch.set_num_interop_threads(min(4, _NUM_CORES))
except RuntimeError:
    pass  # 如果 PyTorch 并行已经初始化则跳过
logger.info(f"系统核心: {_NUM_CORES}, 推理设备: {device if device else "cpu"}")


def _clean_yaml(raw: str) -> str:
    return raw.replace("!!python/tuple", "")


def _list2tuple(d):
    if isinstance(d, dict):
        return {k: _list2tuple(v) for k, v in d.items()}
    elif isinstance(d, list):
        if all(isinstance(x, int) for x in d):
            return tuple(d)
        return [_list2tuple(v) for v in d]
    return d


def load_model(device=None):
    global _model, _config
    if _model is not None:
        return _model, _config
    if device is None:
        device = torch.device("cpu")
    model_path = os.path.join(_MODEL_DIR, _MODEL_NAME)
    config_path = os.path.join(_MODEL_DIR, _CONFIG_NAME)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    with open(config_path, "r") as f:
        raw = f.read()
    config_dict = yaml.safe_load(_clean_yaml(raw))
    config_dict = _list2tuple(config_dict)
    _config = config_dict
    from mel_band_roformer import MelBandRoformer
    model = MelBandRoformer(**config_dict["model"])
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
    logger.info(f"MelBandRoformer 加载成功: {sum(p.numel() for p in model.parameters()):,} params, 设备: {device}")
    _model = model
    return _model, _config


def _auto_batch_size(total_chunks: int) -> int:
    if total_chunks <= 4:
        return total_chunks
    return max(2, min(8, min(cpu_count(), 64), total_chunks))


def separate_vocals(audio_path, device=None):
    import librosa
    if device is None:
        device = torch.device("cpu")
    model, cfg = load_model(device)
    sr = cfg["audio"]["sample_rate"]
    
    audio, orig_sr = librosa.load(audio_path, sr=None, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.ndim == 2 and audio.shape[0] > 2:
        audio = audio.T
    if audio.shape[0] == 1:
        audio = np.concatenate([audio, audio], axis=0)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr, res_type="soxr_hq")
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio = audio / peak * 0.95

    orig_len = audio.shape[1]
    mix_t = torch.from_numpy(audio).float()
    chunk_size = cfg["audio"]["chunk_size"]
    num_overlap = cfg["inference"].get("num_overlap", 1)
    hop_size = chunk_size // 4 if num_overlap == 1 else chunk_size // num_overlap
    mix_len = mix_t.shape[1]

    # ── 短音频：直接推理 ──
    if mix_len <= chunk_size:
        with torch.inference_mode():
            out = model(mix_t.unsqueeze(0).to(device)).cpu()
            if out.dim() == 4:
                out = out[:, 0]
            vocals_np = out.squeeze(0).numpy()
        min_len = min(vocals_np.shape[1], mix_len)
        vocals_np = vocals_np[:, :min_len]
        mix_np = mix_t[:, :min_len].numpy()
        instrumental = mix_np - vocals_np
        return vocals_np, instrumental, sr, f"MelBandRoformer 直接推理 @ {sr}Hz"

    # ── 长音频：填充 → 分块 → 批处理 ──
    pad_needed = hop_size - (mix_len - chunk_size) % hop_size
    mix_padded = torch.cat([
        torch.zeros(2, chunk_size - hop_size),
        mix_t,
        torch.zeros(2, max(0, pad_needed) + chunk_size - hop_size),
    ], dim=1)

    chunks = mix_padded.unfold(1, chunk_size, hop_size).transpose(0, 1).contiguous()
    num_chunks = len(chunks)
    batch_size = _auto_batch_size(num_chunks)
    logger.debug(f"总块数: {num_chunks}, 批大小: {batch_size}")

    result = torch.zeros(2, mix_padded.shape[1])
    counter = torch.zeros(mix_padded.shape[1])
    window = torch.tensor(scipy_signal.windows.hann(chunk_size), dtype=torch.float32)

    with torch.inference_mode():
        for start_idx in range(0, num_chunks, batch_size):
            end_idx = min(start_idx + batch_size, num_chunks)
            batch = chunks[start_idx:end_idx].to(device)
            out = model(batch).cpu()
            if out.dim() == 4:
                out = out[:, 0]
            for j in range(out.shape[0]):
                idx = start_idx + j
                start = idx * hop_size
                seg = out[j]
                actual_len = min(chunk_size, result.shape[1] - start)
                if actual_len <= 0:
                    break
                result[:, start:start+actual_len] += seg[:, :actual_len] * window[:actual_len]
                counter[start:start+actual_len] += window[:actual_len]

    counter = counter.clamp(min=1e-10)
    result = result / counter
    offset = chunk_size - hop_size
    vocals_np = result[:, offset:offset+orig_len].numpy()
    mix_np = mix_t.numpy()
    min_len = min(vocals_np.shape[1], mix_np.shape[1])
    vocals_np = vocals_np[:, :min_len]
    instrumental = mix_np[:, :min_len] - vocals_np
    return vocals_np, instrumental, sr, f"MelBandRoformer 批处理分离 | {num_chunks} chunks x batch {batch_size} @ {sr}Hz"
