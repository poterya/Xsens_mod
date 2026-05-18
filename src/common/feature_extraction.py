"""
Извлечение признаков для детекции состояния винтов.
Используется и в обучении (train_detector.py), и в realtime (realtime_detector.py),
чтобы фичи были идентичны.

Вход: четыре одномерных numpy-массива одинаковой длины
       (total_vibration, rms_x, rms_y, rms_z) — RMS-окна вибрации.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

SAMPLING_HZ = 100.0  # vibration_log.csv пишется ~100 Гц


def _stats_for_axis(values: np.ndarray, prefix: str) -> dict:
    feats: dict = {}
    feats[f"{prefix}_mean"] = float(np.mean(values))
    feats[f"{prefix}_std"] = float(np.std(values))
    feats[f"{prefix}_max"] = float(np.max(values))
    feats[f"{prefix}_min"] = float(np.min(values))
    feats[f"{prefix}_range"] = float(np.ptp(values))
    feats[f"{prefix}_median"] = float(np.median(values))
    q25, q75 = np.percentile(values, [25, 75])
    feats[f"{prefix}_iqr"] = float(q75 - q25)
    feats[f"{prefix}_p90"] = float(np.percentile(values, 90))
    if values.size >= 8 and np.std(values) > 1e-12:
        feats[f"{prefix}_skew"] = float(sps.skew(values, bias=False))
        feats[f"{prefix}_kurtosis"] = float(sps.kurtosis(values, fisher=True, bias=False))
    else:
        feats[f"{prefix}_skew"] = 0.0
        feats[f"{prefix}_kurtosis"] = 0.0
    return feats


def _spectral_features(values: np.ndarray, prefix: str, fs: float = SAMPLING_HZ) -> dict:
    n = values.size
    feats: dict = {
        f"{prefix}_dom_freq": 0.0,
        f"{prefix}_e_0_5": 0.0,
        f"{prefix}_e_5_10": 0.0,
        f"{prefix}_e_10_20": 0.0,
        f"{prefix}_e_20_50": 0.0,
        f"{prefix}_spec_entropy": 0.0,
    }
    if n < 4:
        return feats
    v = values - values.mean()
    if np.std(v) < 1e-12:
        return feats
    spectrum = np.abs(np.fft.rfft(v))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = spectrum * spectrum
    total = float(power.sum()) + 1e-12

    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(power[mask].sum() / total)

    feats[f"{prefix}_e_0_5"] = band(0.0, 5.0)
    feats[f"{prefix}_e_5_10"] = band(5.0, 10.0)
    feats[f"{prefix}_e_10_20"] = band(10.0, 20.0)
    feats[f"{prefix}_e_20_50"] = band(20.0, 50.0)
    if spectrum.size > 1:
        idx = int(np.argmax(spectrum[1:])) + 1
        feats[f"{prefix}_dom_freq"] = float(freqs[idx])
    p = power / total
    p = p[p > 1e-12]
    if p.size:
        feats[f"{prefix}_spec_entropy"] = float(-np.sum(p * np.log(p)))
    return feats


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    sa = float(a.std())
    sb = float(b.std())
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def extract_features(
    total: np.ndarray,
    rms_x: np.ndarray,
    rms_y: np.ndarray,
    rms_z: np.ndarray,
) -> dict:
    feats: dict = {}
    feats.update(_stats_for_axis(total, "total_vibration"))
    feats.update(_stats_for_axis(rms_x, "rms_x"))
    feats.update(_stats_for_axis(rms_y, "rms_y"))
    feats.update(_stats_for_axis(rms_z, "rms_z"))

    total_mean = max(feats["total_vibration_mean"], 1e-9)
    feats["ratio_x_total"] = feats["rms_x_mean"] / total_mean
    feats["ratio_y_total"] = feats["rms_y_mean"] / total_mean
    feats["ratio_z_total"] = feats["rms_z_mean"] / total_mean
    feats["ratio_x_y"] = feats["rms_x_mean"] / max(feats["rms_y_mean"], 1e-9)
    feats["ratio_x_z"] = feats["rms_x_mean"] / max(feats["rms_z_mean"], 1e-9)
    feats["ratio_y_z"] = feats["rms_y_mean"] / max(feats["rms_z_mean"], 1e-9)
    feats["axis_dominance"] = max(
        feats["rms_x_mean"], feats["rms_y_mean"], feats["rms_z_mean"]
    ) / total_mean

    if total.size >= 2:
        diff = np.abs(np.diff(total))
        feats["total_vibration_diff_mean"] = float(np.mean(diff))
        feats["total_vibration_diff_max"] = float(np.max(diff))
        feats["total_vibration_diff_std"] = float(np.std(diff))
    else:
        feats["total_vibration_diff_mean"] = 0.0
        feats["total_vibration_diff_max"] = 0.0
        feats["total_vibration_diff_std"] = 0.0

    feats.update(_spectral_features(total, "total_vibration"))
    feats.update(_spectral_features(rms_x, "rms_x"))
    feats.update(_spectral_features(rms_y, "rms_y"))
    feats.update(_spectral_features(rms_z, "rms_z"))

    feats["corr_xy"] = _safe_corr(rms_x, rms_y)
    feats["corr_xz"] = _safe_corr(rms_x, rms_z)
    feats["corr_yz"] = _safe_corr(rms_y, rms_z)

    feats["asym_xy"] = (feats["rms_x_mean"] - feats["rms_y_mean"]) / total_mean
    feats["asym_xz"] = (feats["rms_x_mean"] - feats["rms_z_mean"]) / total_mean
    feats["asym_yz"] = (feats["rms_y_mean"] - feats["rms_z_mean"]) / total_mean

    return feats
