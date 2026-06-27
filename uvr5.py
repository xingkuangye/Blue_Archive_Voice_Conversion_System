"""
UVR5 人声分离模块
默认使用 Mel-Band Roformer (vocals_mel_band_roformer.ckpt)
回退方案: Demucs
"""
import os
import sys
import torch
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

_device = torch.device("cpu")

# ─── Demucs 缓存 ───
_demucs_model = None
_demucs_model_name = None


# ════════════════════════════════════════════
# 主入口: 人声分离 (MelBandRoformer)
# ════════════════════════════════════════════

def separate_audio_vocals(audio_path, model_name="mel_band_roformer"):
    """
    使用 Mel-Band Roformer 分离人声和背景音

    Args:
        audio_path: 音频文件路径
        model_name: "mel_band_roformer" 或 Demucs 模型名 (htdemucs, htdemucs_ft)

    Returns:
        (vocals, instrumental, status)
    """
    if model_name == "mel_band_roformer":
        return _separate_roformer(audio_path)
    else:
        return _separate_demucs(audio_path, model_name)


def _load_audio_for_sep(audio_path, target_sr=44100):
    """加载音频为标准格式 (channels, samples) float32"""
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
    except Exception as e:
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path, always_2d=True)
            audio = audio.T
        except Exception as e2:
            raise RuntimeError(f"无法加载音频: {e2}")

    if audio.dtype != np.float32:
        if audio.dtype in (np.int16, np.int32):
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype in (np.uint8,):
            audio = audio.astype(np.float32) / 128.0 - 1.0
        else:
            audio = audio.astype(np.float32)

    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.ndim == 2 and audio.shape[0] > 2:
        audio = audio.T
    if audio.shape[0] == 1:
        audio = np.concatenate([audio, audio], axis=0)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")

    audio = np.clip(audio, -1.0, 1.0)
    return audio, target_sr


# ─── Roformer 分离 ───

def _separate_roformer(audio_path):
    """使用 MelBandRoformer 分离"""
    logger.info(f"Roformer 分离: {audio_path}")
    try:
        from vocal_roformer import separate_vocals
        vocals, inst, sr, info = separate_vocals(audio_path, device=_device)
        result_vocals = (sr, vocals)
        result_inst = (sr, inst)
        status = f"Roformer 分离完成 ✅\n人声: {vocals.shape}\n背景音: {inst.shape}"
        return result_vocals, result_inst, status
    except Exception as e:
        logger.error(f"Roformer 分离失败: {e}", exc_info=True)
        return None, None, f"错误: {str(e)}"


# ─── Demucs 分离（回退） ───

def _get_demucs_model(model_name="htdemucs"):
    global _demucs_model, _demucs_model_name
    if _demucs_model is None or _demucs_model_name != model_name:
        try:
            from demucs import pretrained
            logger.info(f"加载 Demucs: {model_name}")
            model = pretrained.get_model(model_name)
            model.to(_device)
            model.eval()
            for p in model.parameters():
                p.data = p.data.float()
            _demucs_model = model
            _demucs_model_name = model_name
        except Exception as e:
            logger.error(f"Demucs 加载失败: {e}")
            return None
    return _demucs_model


def _separate_demucs(audio_path, model_name):
    logger.info(f"Demucs 分离: {audio_path} model={model_name}")
    try:
        from demucs.apply import apply_model
        model = _get_demucs_model(model_name)
        if model is None:
            return None, None, "错误: Demucs 模型加载失败"

        sr = model.samplerate
        audio, _ = _load_audio_for_sep(audio_path, sr)
        tensor = torch.from_numpy(audio).float().unsqueeze(0)

        with torch.no_grad():
            estimate = apply_model(model, tensor, device=_device, shifts=1, split=True, overlap=0.25)
            estimate = estimate.cpu()[0]

        vocals = None
        instrumental = None
        for name, src in zip(model.sources, estimate):
            if name == "vocals":
                vocals = src.numpy()
            else:
                instrumental = (src.numpy() if instrumental is None else instrumental + src.numpy())

        result_vocals = (sr, vocals) if vocals is not None else None
        result_inst = (sr, instrumental) if instrumental is not None else None
        status = f"Demucs 分离完成 ✅\n人声: {result_vocals[1].shape}" if result_vocals is not None else "完成"
        return result_vocals, result_inst, status

    except Exception as e:
        logger.error(f"Demucs 分离失败: {e}", exc_info=True)
        return None, None, f"错误: {str(e)}"


# ─── 去混响（保持不变） ───

def separate_dereverb(audio_path, overlap=4):
    """使用 MDX23C De-Reverb 消除混响"""
    logger.info(f"MDX23C 去混响: {audio_path}")
    try:
        from mdx23c_dereverb import process as mdx23c_process
        dry, reverb, sr = mdx23c_process(audio_path, overlap=overlap, device=_device)
        result_dry = (sr, dry)
        result_reverb = (sr, reverb)
        status = f"去混响完成 ✅\n干声: {dry.shape}\n混响: {reverb.shape}"
        return result_dry, result_reverb, status
    except FileNotFoundError as e:
        return None, None, f"错误: {str(e)}"
    except Exception as e:
        logger.error(f"去混响失败: {e}", exc_info=True)
        return None, None, f"错误: {str(e)}"
