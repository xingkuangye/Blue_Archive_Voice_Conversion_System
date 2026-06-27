"""
变声转换逻辑 — 封装 RVC pipeline
"""
import os
import json
import asyncio
import logging
import traceback
import numpy as np
import librosa
import edge_tts
from datetime import datetime

logger = logging.getLogger(__name__)


async def load_tts_voices():
    """加载 Edge-TTS 语音列表"""
    try:
        voice_list = await edge_tts.list_voices()
        voices = [
            {"name": f"{v['ShortName']}-{v['Gender']}", "short_name": v["ShortName"], "gender": v["Gender"], "locale": v["Locale"]}
            for v in voice_list
        ]
        return voices
    except Exception as e:
        logger.warning(f"Failed to load TTS voices: {e}")
        return [
            {"name": "zh-CN-XiaoxiaoNeural-Female", "short_name": "zh-CN-XiaoxiaoNeural", "gender": "Female", "locale": "zh-CN"},
            {"name": "zh-CN-YunxiNeural-Male", "short_name": "zh-CN-YunxiNeural", "gender": "Male", "locale": "zh-CN"},
            {"name": "en-US-AnaNeural-Female", "short_name": "en-US-AnaNeural", "gender": "Female", "locale": "en-US"},
        ]


def load_audio_from_path(path: str, target_sr: int = 16000):
    """从文件路径加载音频"""
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


async def generate_tts_audio(text: str, voice: str) -> str:
    """生成 TTS 音频，返回临时文件路径"""
    temp_file = "temp_tts_output.mp3"
    short_name = "-".join(voice.split("-")[:-1])
    await edge_tts.Communicate(text, short_name).save(temp_file)
    return temp_file


def run_convert(
    char_obj: dict,
    audio_input_path: str,
    f0_up_key: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.7,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 1.0,
    protect: float = 0.5,
    output_path: str = "temp_output.wav",
):
    """
    执行变声转换

    Returns:
        (output_path, sample_rate, info_string)
    """
    import sys
    from backend.model_manager import hubert_model

    net_g = char_obj["net_g"]
    vc = char_obj["vc"]
    tgt_sr = char_obj["tgt_sr"]
    if_f0 = char_obj["if_f0"]
    version = char_obj["version"]
    file_index = char_obj["index_path"]

    # 加载音频
    audio, sr = load_audio_from_path(audio_input_path)
    logger.info(f"Audio loaded: sr={sr}, duration={len(audio)/sr:.2f}s")

    times = [0, 0, 0]
    f0_up_key = int(f0_up_key)

    audio_opt = vc.pipeline(
        hubert_model, net_g, 0, audio, audio_input_path,
        times, f0_up_key, f0_method, file_index, index_rate,
        if_f0, filter_radius, tgt_sr, resample_sr,
        rms_mix_rate, version, protect, f0_file=None,
    )

    info = (
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"npy: {times[0]:.2f}s | f0: {times[1]:.2f}s | infer: {times[2]:.2f}s"
    )

    # 保存为 wav
    import soundfile as sf
    sf.write(output_path, audio_opt, tgt_sr)
    logger.info(f"Output saved: {output_path}")

    return output_path, tgt_sr, info
