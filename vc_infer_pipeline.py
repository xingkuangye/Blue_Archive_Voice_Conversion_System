"""
VC Infer Pipeline — 翻唱变声管线
原版 _vc_single 逻辑，pipeline 入口不变
"""
import numpy as np, parselmouth, torch, pdb, sys, os
from time import time as ttime
import torch.nn.functional as F
import scipy.signal as signal
import pyworld, os, traceback, faiss, librosa, torchcrepe
from scipy import signal
from functools import lru_cache
import tempfile, shutil

os.environ["PYTHONIOENCODING"] = "utf-8"
now_dir = os.getcwd()
sys.path.append(now_dir)
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)
_input_audio_path2wav = {}


def read_faiss_index_with_chinese_path(file_index):
    try:
        return faiss.read_index(file_index)
    except Exception:
        try:
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, os.path.basename(file_index))
            shutil.copy2(file_index, temp_file)
            idx = faiss.read_index(temp_file)
            os.remove(temp_file)
            return idx
        except:
            return None, None


@lru_cache
def _cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = _input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(audio, fs=fs, f0_ceil=f0max, f0_floor=f0min, frame_period=frame_period)
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(rms1.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(rms2.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (torch.pow(rms1, torch.tensor(1 - rate)) * torch.pow(rms2, torch.tensor(rate - 1))).numpy()
    return data2


class VC(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad, config.x_query, config.x_center, config.x_max, config.is_half,
        )
        self.sr = 16000
        self.window = 160
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max
        self.device = config.device

    def get_f0(self, input_audio_path, x, p_len, f0_up_key, f0_method, filter_radius, inp_f0=None):
        global _input_audio_path2wav
        time_step = self.window / self.sr * 1000
        f0_min = 50; f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        if f0_method == "pm":
            f0 = parselmouth.Sound(x, self.sr).to_pitch_ac(
                time_step=time_step / 1000, voicing_threshold=0.6,
                pitch_floor=f0_min, pitch_ceiling=f0_max,
            ).selected_array["frequency"]
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        elif f0_method == "harvest":
            _input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = _cache_harvest_f0(input_audio_path, self.sr, f0_max, f0_min, 10)
            if filter_radius > 2: f0 = signal.medfilt(f0, 3)
        elif f0_method == "crepe":
            audio_t = torch.tensor(np.copy(x))[None].float()
            f0, pd = torchcrepe.predict(audio_t, self.sr, self.window, f0_min, f0_max, "full", batch_size=512, device=self.device, return_periodicity=True)
            pd = torchcrepe.filter.median(pd, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[pd < 0.1] = 0; f0 = f0[0].cpu().numpy()
        elif f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                from rmvpe import RMVPE
                self.model_rmvpe = RMVPE("rmvpe.pt", is_half=self.is_half, device=self.device)
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        f0 *= pow(2, f0_up_key / 12)
        tf0 = self.sr // self.window
        if inp_f0 is not None:
            delta_t = np.round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1).astype("int16")
            replace_f0 = np.interp(list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1])
            shape = f0[self.x_pad * tf0: self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0: self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)
        return f0_coarse, f0bak

    # ═══════ 管线性入口 ═══════

    def pipeline(self, model, net_g, sid, audio, input_audio_path, times,
                 f0_up_key, f0_method, file_index, index_rate, if_f0,
                 filter_radius, tgt_sr, resample_sr, rms_mix_rate, version, protect,
                 f0_file=None,):
        if file_index != "" and os.path.exists(file_index) and index_rate != 0:
            try:
                idx_result = read_faiss_index_with_chinese_path(file_index)
                index = big_npy = (None, None) if isinstance(idx_result, tuple) else (idx_result, idx_result.reconstruct_n(0, idx_result.ntotal))
                index, big_npy = index
            except:
                traceback.print_exc(); index = big_npy = None
        else:
            index = big_npy = None

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")

        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i: i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t - self.t_query + np.where(
                        np.abs(audio_sum[t - self.t_query: t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query: t + self.t_query]).min()
                    )[0][0]
                )

        if not opt_ts:
            return self._pipeline_short(model, net_g, sid, audio, input_audio_path, times,
                                        f0_up_key, f0_method, index, big_npy, index_rate,
                                        if_f0, filter_radius, tgt_sr, resample_sr,
                                        rms_mix_rate, version, protect, f0_file)

        return self._pipeline_segmented(model, net_g, sid, audio, input_audio_path, times,
                                        f0_up_key, f0_method, index, big_npy, index_rate,
                                        if_f0, filter_radius, tgt_sr, resample_sr,
                                        rms_mix_rate, version, protect, f0_file, opt_ts)

    # ═══════ 不分段 ═══════

    def _pipeline_short(self, model, net_g, sid, audio, input_audio_path, times,
                        f0_up_key, f0_method, index, big_npy, index_rate,
                        if_f0, filter_radius, tgt_sr, resample_sr,
                        rms_mix_rate, version, protect, f0_file):
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        sid_t = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch = pitchf = None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(input_audio_path, audio_pad, p_len, f0_up_key, f0_method, filter_radius)
            pitch = pitch[:p_len]; pitchf = pitchf[:p_len]
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        audio_opt = self._vc_single(model, net_g, sid_t, audio_pad, pitch, pitchf,
                                    times, index, big_npy, index_rate, version, protect)
        audio_opt = audio_opt[self.t_pad_tgt:-self.t_pad_tgt] if self.t_pad_tgt > 0 else audio_opt
        return self._postprocess(audio_opt, audio, tgt_sr, resample_sr, rms_mix_rate)

    # ═══════ 分段（逐段 _vc_single） ═══════

    def _pipeline_segmented(self, model, net_g, sid, audio, input_audio_path, times,
                            f0_up_key, f0_method, index, big_npy, index_rate,
                            if_f0, filter_radius, tgt_sr, resample_sr,
                            rms_mix_rate, version, protect, f0_file, opt_ts):
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        sid_t = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        t1 = ttime()
        p_len = audio_pad.shape[0] // self.window
        pitch = pitchf = None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(input_audio_path, audio_pad, p_len, f0_up_key, f0_method, filter_radius)
            pitch = pitch[:p_len]; pitchf = pitchf[:p_len]
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        t2 = ttime(); times[1] += t2 - t1

        audio_opt = []
        s = 0
        for t in opt_ts:
            t = t // self.window * self.window
            seg = audio_pad[s : t + self.t_pad2 + self.window]
            sp = pitch[:, s // self.window : (t + self.t_pad2) // self.window] if if_f0 == 1 else None
            sf = pitchf[:, s // self.window : (t + self.t_pad2) // self.window] if if_f0 == 1 else None
            out = self._vc_single(model, net_g, sid_t, seg, sp, sf,
                                  times, index, big_npy, index_rate, version, protect)
            trim = self.t_pad_tgt
            audio_opt.append(out[trim:-trim] if trim > 0 else out)
            s = t

        seg = audio_pad[t:] if t is not None else audio_pad
        sp = pitch[:, t // self.window :] if (t is not None and if_f0 == 1) else (pitch if if_f0 == 1 else None)
        sf = pitchf[:, t // self.window :] if (t is not None and if_f0 == 1) else (pitchf if if_f0 == 1 else None)
        out = self._vc_single(model, net_g, sid_t, seg, sp, sf,
                              times, index, big_npy, index_rate, version, protect)
        trim = self.t_pad_tgt
        audio_opt.append(out[trim:-trim] if trim > 0 else out)

        audio_opt = np.concatenate(audio_opt)
        del pitch, pitchf, sid_t
        return self._postprocess(audio_opt, audio, tgt_sr, resample_sr, rms_mix_rate)

    def _postprocess(self, audio_opt, orig_audio, tgt_sr, resample_sr, rms_mix_rate):
        if rms_mix_rate != 1:
            audio_opt = change_rms(orig_audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=resample_sr)
        max_int16 = 32768 / max(1, np.abs(audio_opt).max() / 0.99)
        return (audio_opt * max_int16).astype(np.int16)

    # ═══════ 单段 vc（原始逻辑，无改动） ═══════

    def _vc_single(self, model, net_g, sid, audio0, pitch, pitchf,
                   times, index, big_npy, index_rate, version, protect):
        feats = torch.from_numpy(audio0)
        feats = feats.half() if self.is_half else feats.float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        feats = feats.view(1, -1).to(self.device)
        mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(source=feats, padding_mask=mask, output_layer=9 if version == "v1" else 12)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        if index is not None and big_npy is not None and index_rate != 0:
            npy = feats[0].cpu().numpy()
            if self.is_half: npy = npy.astype("float32")
            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            if self.is_half: npy = npy.astype("float16")
            feats = torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]; pitchf = pitchf[:, :p_len]
        if protect < 0.5 and pitch is not None and pitchf is not None:
            pff = pitchf.clone()
            pff[pitchf > 0] = 1; pff[pitchf < 1] = protect
            feats = feats * pff.unsqueeze(-1) + feats0 * (1 - pff.unsqueeze(-1))
        p_len_t = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            if pitch is not None and pitchf is not None:
                out = net_g.infer(feats, p_len_t, pitch, pitchf, sid)[0][0, 0].data.cpu().float().numpy()
            else:
                out = net_g.infer(feats, p_len_t, sid)[0][0, 0].data.cpu().float().numpy()
        del feats, p_len_t, mask
        t2 = ttime(); times[0] += t1 - t0; times[2] += t2 - t1
        return out

    # ═══════ 旧版接口兼容 ═══════

    def vc(self, model, net_g, sid, audio0, pitch, pitchf,
           times, index, big_npy, index_rate, version, protect):
        return self._vc_single(model, net_g, torch.tensor(sid, device=self.device).unsqueeze(0).long(),
                               audio0, pitch, pitchf, times, index, big_npy, index_rate, version, protect)
