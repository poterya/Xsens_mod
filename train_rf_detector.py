#!/usr/bin/env python3
"""
Обучение Random Forest для детекции поломки винта квадрокоптера
по вибрационным данным Xsens.

Дополнительно к статистическим признакам извлекаются FFT-признаки
(доминантная частота, спектральная энергия, энтропия).

Данные:
  Normal_mod/   — полёты с целыми винтами
  Deformed_mod/ — полёты со сломанным передним левым винтом

Результат:
  propeller_fault_rf.pkl — обученная модель RandomForest
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix


WINDOW_SIZE = 50
SAMPLE_RATE_HZ = 100


def _fft_features(signal: np.ndarray, prefix: str) -> dict:
    """Спектральные признаки одного канала."""
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal - np.mean(signal)))
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE_HZ)

    power = fft_vals ** 2
    total_power = np.sum(power) + 1e-12

    peak_idx = np.argmax(fft_vals[1:]) + 1
    dominant_freq = freqs[peak_idx]
    dominant_mag = fft_vals[peak_idx]

    p = power / total_power
    spectral_entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))

    spectral_centroid = np.sum(freqs * power) / total_power

    return {
        f"{prefix}_dom_freq": dominant_freq,
        f"{prefix}_dom_mag": dominant_mag,
        f"{prefix}_spectral_energy": total_power,
        f"{prefix}_spectral_entropy": spectral_entropy,
        f"{prefix}_spectral_centroid": spectral_centroid,
    }


def extract_features_from_window(window: pd.DataFrame) -> dict:
    feats: dict = {}

    for col in ("total_vibration", "rms_x", "rms_y", "rms_z"):
        v = window[col].values
        feats[f"{col}_mean"] = np.mean(v)
        feats[f"{col}_std"] = np.std(v)
        feats[f"{col}_max"] = np.max(v)
        feats[f"{col}_min"] = np.min(v)
        feats[f"{col}_range"] = np.ptp(v)
        feats[f"{col}_median"] = np.median(v)
        feats[f"{col}_q25"] = np.percentile(v, 25)
        feats[f"{col}_q75"] = np.percentile(v, 75)
        feats[f"{col}_iqr"] = feats[f"{col}_q75"] - feats[f"{col}_q25"]
        feats[f"{col}_skew"] = float(pd.Series(v).skew())
        feats[f"{col}_kurtosis"] = float(pd.Series(v).kurtosis())
        feats.update(_fft_features(v, col))

    t_mean = max(feats["total_vibration_mean"], 1e-9)
    feats["ratio_x_total"] = feats["rms_x_mean"] / t_mean
    feats["ratio_y_total"] = feats["rms_y_mean"] / t_mean
    feats["ratio_z_total"] = feats["rms_z_mean"] / t_mean

    total = window["total_vibration"].values
    if len(total) >= 2:
        d = np.abs(np.diff(total))
        feats["total_vibration_diff_mean"] = np.mean(d)
        feats["total_vibration_diff_max"] = np.max(d)
        feats["total_vibration_diff_std"] = np.std(d)
    else:
        feats["total_vibration_diff_mean"] = 0.0
        feats["total_vibration_diff_max"] = 0.0
        feats["total_vibration_diff_std"] = 0.0

    feats["xy_correlation"] = float(
        np.corrcoef(window["rms_x"].values, window["rms_y"].values)[0, 1]
    )
    feats["xz_correlation"] = float(
        np.corrcoef(window["rms_x"].values, window["rms_z"].values)[0, 1]
    )
    feats["yz_correlation"] = float(
        np.corrcoef(window["rms_y"].values, window["rms_z"].values)[0, 1]
    )

    return feats


def load_session(csv_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    needed = {"total_vibration", "rms_x", "rms_y", "rms_z"}
    if not needed.issubset(df.columns):
        return None
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=list(needed), inplace=True)
    if len(df) < WINDOW_SIZE:
        return None
    return df


def extract_session_features(df: pd.DataFrame, label: int, step: int = 25) -> list[dict]:
    rows: list[dict] = []
    for start in range(0, len(df) - WINDOW_SIZE + 1, step):
        window = df.iloc[start : start + WINDOW_SIZE]
        feats = extract_features_from_window(window)
        feats["label"] = label
        rows.append(feats)
    return rows


def build_dataset(base: Path) -> pd.DataFrame:
    all_rows: list[dict] = []

    for csv_path in sorted((base / "Normal_mod").rglob("vibration_log.csv")):
        df = load_session(csv_path)
        if df is None:
            continue
        all_rows.extend(extract_session_features(df, label=0))

    for csv_path in sorted((base / "Deformed_mod").rglob("vibration_log.csv")):
        df = load_session(csv_path)
        if df is None:
            continue
        all_rows.extend(extract_session_features(df, label=1))

    if not all_rows:
        print("Нет данных для обучения!", file=sys.stderr)
        sys.exit(1)

    return pd.DataFrame(all_rows)


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Random Forest детектор поломки винта")
    parser.add_argument("--base", type=Path, default=base)
    parser.add_argument("--n-trees", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--output", type=str, default="propeller_fault_rf.pkl")
    args = parser.parse_args()

    print("Сбор признаков (статистика + FFT) из Normal_mod и Deformed_mod …")
    dataset = build_dataset(args.base)

    nan_cols = dataset.columns[dataset.isna().any()].tolist()
    if nan_cols:
        print(f"NaN в столбцах: {nan_cols} — заполняю нулями")
        dataset.fillna(0.0, inplace=True)

    feature_cols = [c for c in dataset.columns if c != "label"]
    X = dataset[feature_cols].values
    y = dataset["label"].values

    n_normal = int(np.sum(y == 0))
    n_deformed = int(np.sum(y == 1))
    print(f"Окон: Normal={n_normal}, Deformed={n_deformed}, всего={len(y)}")
    print(f"Признаков: {len(feature_cols)}")

    clf = RandomForestClassifier(
        n_estimators=args.n_trees,
        max_depth=args.max_depth,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = cross_val_score(clf, X, y, cv=skf, scoring="f1")
    acc_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    print(f"\nCross-validation F1:       {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    print(f"Cross-validation Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")

    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("\n=== Отчёт на полной выборке ===")
    print(classification_report(y, y_pred, target_names=["Normal", "Deformed"]))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))

    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print("Топ-15 признаков:")
    for i in top_idx:
        print(f"  {feature_cols[i]:40s} {importances[i]:.4f}")

    model_path = args.base / args.output
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "classifier": clf,
                "feature_names": feature_cols,
                "window_size": WINDOW_SIZE,
                "model_type": "RandomForest",
            },
            f,
        )
    print(f"\nМодель сохранена: {model_path}")
    print(f"Деревьев: {args.n_trees}, макс. глубина: {args.max_depth}")


if __name__ == "__main__":
    main()
