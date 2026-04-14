#!/usr/bin/env python3
"""
Обучение дерева решений для детекции поломки винта квадрокоптера
по вибрационным данным Xsens.

Данные:
  Normal_mod/   — полёты с целыми винтами
  Deformed_mod/ — полёты со сломанным передним левым винтом

Результат:
  propeller_fault_model.pkl — обученная модель + scaler
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix


WINDOW_SIZE = 50


def extract_features_from_window(window: pd.DataFrame) -> dict:
    """Извлекает признаки из одного скользящего окна (WINDOW_SIZE строк)."""
    feats: dict = {}
    for col in ("total_vibration", "rms_x", "rms_y", "rms_z"):
        v = window[col].values
        feats[f"{col}_mean"] = np.mean(v)
        feats[f"{col}_std"] = np.std(v)
        feats[f"{col}_max"] = np.max(v)
        feats[f"{col}_min"] = np.min(v)
        feats[f"{col}_range"] = np.ptp(v)
        feats[f"{col}_median"] = np.median(v)

    feats["ratio_x_total"] = feats["rms_x_mean"] / max(feats["total_vibration_mean"], 1e-9)
    feats["ratio_y_total"] = feats["rms_y_mean"] / max(feats["total_vibration_mean"], 1e-9)
    feats["ratio_z_total"] = feats["rms_z_mean"] / max(feats["total_vibration_mean"], 1e-9)

    total = window["total_vibration"].values
    if len(total) >= 2:
        feats["total_vibration_diff_mean"] = np.mean(np.abs(np.diff(total)))
        feats["total_vibration_diff_max"] = np.max(np.abs(np.diff(total)))
    else:
        feats["total_vibration_diff_mean"] = 0.0
        feats["total_vibration_diff_max"] = 0.0

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
    normal_dir = base / "Normal_mod"
    deformed_dir = base / "Deformed_mod"

    for csv_path in sorted(normal_dir.rglob("vibration_log.csv")):
        df = load_session(csv_path)
        if df is None:
            continue
        all_rows.extend(extract_session_features(df, label=0))

    for csv_path in sorted(deformed_dir.rglob("vibration_log.csv")):
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
    parser = argparse.ArgumentParser(description="Обучение детектора поломки винта")
    parser.add_argument("--base", type=Path, default=base)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--output", type=str, default="propeller_fault_model.pkl")
    args = parser.parse_args()

    print("Сбор признаков из Normal_mod и Deformed_mod …")
    dataset = build_dataset(args.base)
    feature_cols = [c for c in dataset.columns if c != "label"]
    X = dataset[feature_cols].values
    y = dataset["label"].values

    n_normal = int(np.sum(y == 0))
    n_deformed = int(np.sum(y == 1))
    print(f"Окон: Normal={n_normal}, Deformed={n_deformed}, всего={len(y)}")

    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf, scoring="f1")
    print(f"\nCross-validation F1: {scores.mean():.4f} ± {scores.std():.4f}")
    acc_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    print(f"Cross-validation Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")

    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("\n=== Отчёт на полной выборке ===")
    print(classification_report(y, y_pred, target_names=["Normal", "Deformed"]))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))

    print(f"\nГлубина дерева: {clf.get_depth()}, листьев: {clf.get_n_leaves()}")
    print("\nДерево решений:")
    print(export_text(clf, feature_names=feature_cols, max_depth=4))

    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("Топ-10 признаков:")
    for i in top_idx:
        print(f"  {feature_cols[i]:35s} {importances[i]:.4f}")

    model_path = args.base / args.output
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "classifier": clf,
                "feature_names": feature_cols,
                "window_size": WINDOW_SIZE,
            },
            f,
        )
    print(f"\nМодель сохранена: {model_path}")


if __name__ == "__main__":
    main()
