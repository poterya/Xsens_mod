"""Модуль формирования признаков из вибрационных окон.

Окно: 10 отсчётов (0.1 с при 100 Гц), шаг: 5 отсчётов (перекрытие 50%).
Метка класса определяется по папке: Normal -> 0 (норма), Deformed -> 1 (поломка).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_SIZE = 10
STEP = 5
SAMPLE_RATE_HZ = 100.0
SPLITS = ("train", "val", "test")
CLASS_DIRS = {"Normal": 0, "Deformed": 1}
LABEL_NAME = {0: "норма", 1: "поломка"}

FFT_BANDS = (
    (0.0, 10.0),
    (10.0, 20.0),
    (20.0, 35.0),
    (35.0, 50.0),
)

REQUIRED_COLUMNS = ("time_seconds", "total_vibration", "rms_x", "rms_y", "rms_z")


@dataclass
class WindowSet:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list[str]


def _stats_block(values: np.ndarray, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    out[f"{prefix}_mean"] = float(np.mean(values))
    out[f"{prefix}_std"] = float(np.std(values))
    out[f"{prefix}_max"] = float(np.max(values))
    out[f"{prefix}_min"] = float(np.min(values))
    out[f"{prefix}_p95"] = float(np.percentile(values, 95))
    out[f"{prefix}_range"] = float(np.ptp(values))
    return out


def _fft_bands(values: np.ndarray, prefix: str) -> dict[str, float]:
    n = values.size
    spectrum = np.fft.rfft(values - float(np.mean(values)))
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE_HZ)
    power = (np.abs(spectrum) ** 2) / max(1, n)
    out: dict[str, float] = {}
    for lo, hi in FFT_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        out[f"{prefix}_band_{int(lo)}_{int(hi)}"] = float(np.sum(power[mask]))
    return out


def _corrs(total: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, float]:
    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    return {
        "corr_x_y": safe_corr(x, y),
        "corr_x_z": safe_corr(x, z),
        "corr_y_z": safe_corr(y, z),
        "corr_total_x": safe_corr(total, x),
        "corr_total_y": safe_corr(total, y),
        "corr_total_z": safe_corr(total, z),
    }


def extract_features(total: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats.update(_stats_block(total, "total"))
    feats.update(_stats_block(x, "rms_x"))
    feats.update(_stats_block(y, "rms_y"))
    feats.update(_stats_block(z, "rms_z"))
    feats.update(_fft_bands(total, "spec_total"))
    feats.update(_fft_bands(x, "spec_x"))
    feats.update(_fft_bands(y, "spec_y"))
    feats.update(_fft_bands(z, "spec_z"))
    feats.update(_corrs(total, x, y, z))
    return feats


def _read_flight(csv_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if not set(REQUIRED_COLUMNS).issubset(df.columns):
        return None
    for c in REQUIRED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    if len(df) < WINDOW_SIZE:
        return None
    return df


def _windows_from_flight(df: pd.DataFrame, label: int, flight_id: str) -> list[dict]:
    total = df["total_vibration"].to_numpy(dtype=float)
    rx = df["rms_x"].to_numpy(dtype=float)
    ry = df["rms_y"].to_numpy(dtype=float)
    rz = df["rms_z"].to_numpy(dtype=float)
    rows: list[dict] = []
    end = len(df) - WINDOW_SIZE + 1
    for start in range(0, end, STEP):
        sl = slice(start, start + WINDOW_SIZE)
        feats = extract_features(total[sl], rx[sl], ry[sl], rz[sl])
        feats["label"] = label
        feats["flight_id"] = flight_id
        rows.append(feats)
    return rows


def build_split(dataset_root: Path, split: str) -> WindowSet:
    rows: list[dict] = []
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise SystemExit(f"Нет каталога {split_dir}")
    for class_dir, label in CLASS_DIRS.items():
        cls_root = split_dir / class_dir
        if not cls_root.is_dir():
            continue
        for csv_path in sorted(cls_root.rglob("vibration_log.csv")):
            df = _read_flight(csv_path)
            if df is None:
                continue
            flight_id = csv_path.parent.name
            rows.extend(_windows_from_flight(df, label, flight_id))
    if not rows:
        raise SystemExit(f"Нет окон в {split_dir}")
    df_rows = pd.DataFrame(rows)
    feat_cols = [c for c in df_rows.columns if c not in ("label", "flight_id")]
    X = df_rows[feat_cols].to_numpy(dtype=float)
    y = df_rows["label"].to_numpy(dtype=int)
    groups = df_rows["flight_id"].to_numpy()
    return WindowSet(X=X, y=y, groups=groups, feature_names=feat_cols)


def build_all(dataset_root: Path) -> dict[str, WindowSet]:
    return {split: build_split(dataset_root, split) for split in SPLITS}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "dataset"
    data = build_all(root)
    for split, ws in data.items():
        n_pos = int((ws.y == 1).sum())
        n_neg = int((ws.y == 0).sum())
        print(f"{split:5s} окон={ws.X.shape[0]:>6d}  признаков={ws.X.shape[1]:>3d}  норма={n_neg}  поломка={n_pos}")
