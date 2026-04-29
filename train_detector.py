#!/usr/bin/env python3
"""
Обучение дерева решений для детекции состояния винтов квадрокоптера
по вибрационным данным Xsens.

Классы (multiclass):
  normal       — все винты целые
  front_left   — повреждён передний левый (мотор B)
  front_right  — повреждён передний правый (мотор A)
  rear_left    — повреждён задний левый  (мотор A)
  rear_right   — повреждён задний правый (мотор B)

Источник данных по умолчанию: datasets/set_02
  Normal_mod/<sim>/vibration_log.csv
  Deformed_mod/<position>/<sim>/vibration_log.csv

Результат:
  propeller_fault_model.pkl — обученная модель + список признаков + классы
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

WINDOW_SIZE = 50

CLASS_LABELS: tuple[str, ...] = (
    "normal",
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
)

POSITION_RU: dict[str, str] = {
    "normal": "норма",
    "front_left": "передний левый",
    "front_right": "передний правый",
    "rear_left": "задний левый",
    "rear_right": "задний правый",
}


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


def extract_features_from_window(window: pd.DataFrame) -> dict:
    """Признаки одного скользящего окна (WINDOW_SIZE строк)."""
    feats: dict = {}
    cols = ("total_vibration", "rms_x", "rms_y", "rms_z")
    for col in cols:
        feats.update(_stats_for_axis(window[col].values, col))

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

    total = window["total_vibration"].values
    if len(total) >= 2:
        diff = np.abs(np.diff(total))
        feats["total_vibration_diff_mean"] = float(np.mean(diff))
        feats["total_vibration_diff_max"] = float(np.max(diff))
        feats["total_vibration_diff_std"] = float(np.std(diff))
    else:
        feats["total_vibration_diff_mean"] = 0.0
        feats["total_vibration_diff_max"] = 0.0
        feats["total_vibration_diff_std"] = 0.0

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


def extract_session_features(
    df: pd.DataFrame, label: int, step: int = 25
) -> list[dict]:
    rows: list[dict] = []
    for start in range(0, len(df) - WINDOW_SIZE + 1, step):
        window = df.iloc[start : start + WINDOW_SIZE]
        feats = extract_features_from_window(window)
        feats["label"] = label
        rows.append(feats)
    return rows


def build_dataset(base: Path) -> pd.DataFrame:
    """Из base/Normal_mod и base/Deformed_mod/<position> делает таблицу окон."""
    label_to_idx = {name: i for i, name in enumerate(CLASS_LABELS)}

    all_rows: list[dict] = []
    counts: dict[str, int] = {n: 0 for n in CLASS_LABELS}

    normal_dir = base / "Normal_mod"
    if normal_dir.is_dir():
        for csv_path in sorted(normal_dir.rglob("vibration_log.csv")):
            df = load_session(csv_path)
            if df is None:
                continue
            rows = extract_session_features(df, label=label_to_idx["normal"])
            all_rows.extend(rows)
            counts["normal"] += 1

    deformed_dir = base / "Deformed_mod"
    if deformed_dir.is_dir():
        for position in CLASS_LABELS[1:]:
            pos_dir = deformed_dir / position
            if not pos_dir.is_dir():
                continue
            for csv_path in sorted(pos_dir.rglob("vibration_log.csv")):
                df = load_session(csv_path)
                if df is None:
                    continue
                rows = extract_session_features(df, label=label_to_idx[position])
                all_rows.extend(rows)
                counts[position] += 1

    if not all_rows:
        print(f"Нет данных для обучения в {base}", file=sys.stderr)
        sys.exit(1)

    print("Сессий по классам:")
    for name in CLASS_LABELS:
        print(f"  {name:12s} {counts[name]}")

    return pd.DataFrame(all_rows)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Обучение детектора (multi-class: normal + 4 позиции винта)"
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=here / "datasets" / "set_02",
        help="Каталог с Normal_mod/ и Deformed_mod/<position>/",
    )
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.95,
        help="Минимально приемлемая средняя cross-val accuracy",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="propeller_fault_model.pkl",
        help="Имя файла модели (сохраняется рядом с этим скриптом)",
    )
    args = parser.parse_args()

    base = args.base.resolve()
    if not base.is_dir():
        print(f"База данных не найдена: {base}", file=sys.stderr)
        sys.exit(1)

    print(f"Источник: {base}")
    print("Сбор признаков из Normal_mod и Deformed_mod/<position> …")
    dataset = build_dataset(base)
    feature_cols = [c for c in dataset.columns if c != "label"]
    X = dataset[feature_cols].values
    y = dataset["label"].values

    print("\nОкон по классам:")
    for i, name in enumerate(CLASS_LABELS):
        n = int(np.sum(y == i))
        print(f"  {name:12s} {n}")
    print(f"  всего окон:  {len(y)}")

    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=42,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    f1_scores = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")
    print(
        f"\nCross-validation accuracy : {acc_scores.mean():.4f} ± {acc_scores.std():.4f}"
    )
    print(f"Cross-validation f1_macro : {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

    if acc_scores.mean() < args.target_accuracy:
        print(
            f"\n[!] Точность {acc_scores.mean():.4f} ниже целевой "
            f"{args.target_accuracy:.2f}. Попробуйте --max-depth или больше данных.",
            file=sys.stderr,
        )

    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("\n=== Отчёт на полной выборке ===")
    print(classification_report(y, y_pred, target_names=list(CLASS_LABELS)))
    print("Confusion matrix (rows=truth, cols=pred):")
    print("    " + " ".join(f"{lbl:>11s}" for lbl in CLASS_LABELS))
    cm = confusion_matrix(y, y_pred, labels=list(range(len(CLASS_LABELS))))
    for lbl, row in zip(CLASS_LABELS, cm):
        print(f"{lbl:12s} " + " ".join(f"{v:>11d}" for v in row))

    print(f"\nГлубина дерева: {clf.get_depth()}, листьев: {clf.get_n_leaves()}")
    print("\nДерево решений (top уровни):")
    print(export_text(clf, feature_names=feature_cols, max_depth=4))

    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:12]
    print("Топ-12 признаков:")
    for i in top_idx:
        print(f"  {feature_cols[i]:32s} {importances[i]:.4f}")

    model_path = here / args.output
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "classifier": clf,
                "feature_names": feature_cols,
                "class_labels": list(CLASS_LABELS),
                "position_ru": POSITION_RU,
                "window_size": WINDOW_SIZE,
                "target_accuracy": args.target_accuracy,
                "cv_accuracy": float(acc_scores.mean()),
            },
            f,
        )
    print(f"\nМодель сохранена: {model_path}")
    print(
        f"Cross-val accuracy: {acc_scores.mean():.4f}; "
        f"target: {args.target_accuracy:.2f}"
    )


if __name__ == "__main__":
    main()
