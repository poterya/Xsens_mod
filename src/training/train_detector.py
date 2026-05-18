#!/usr/bin/env python3
"""
Обучение классификатора состояния винтов квадрокоптера по вибрационным данным.

Классы (multiclass, по умолчанию):
  normal       — все винты целые
  front_left   — повреждён передний левый (мотор B)
  front_right  — повреждён передний правый (мотор A)
  rear_left    — повреждён задний левый  (мотор A)
  rear_right   — повреждён задний правый (мотор B)

Бинарный режим (--binary):
  normal — все винты целые
  fault  — любая поломка (front_left | front_right | rear_left | rear_right)

Особенности:
- Расширенные признаки: спектр (FFT) + корреляции/асимметрии между осями
  (см. feature_extraction.extract_features).
- Честная валидация GroupKFold по сессиям (без утечки соседних окон).
- RandomForest по умолчанию; --model tree — одно решающее дерево.
"""
from __future__ import annotations

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from common.feature_extraction import extract_features

WINDOW_SIZE = 50
STEP = 10

CLASS_LABELS_FULL: tuple[str, ...] = (
    "normal",
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
)
CLASS_LABELS_BINARY: tuple[str, ...] = ("normal", "fault")

POSITION_RU: dict[str, str] = {
    "normal": "норма",
    "front_left": "передний левый",
    "front_right": "передний правый",
    "rear_left": "задний левый",
    "rear_right": "задний правый",
    "fault": "поломка",
}

# Группа мотора в X-конфигурации: A — front_right + rear_left, B — front_left + rear_right.
PROP_GROUP: dict[str, str] = {
    "front_left": "B",
    "front_right": "A",
    "rear_left": "A",
    "rear_right": "B",
    "normal": "-",
}


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
    df: pd.DataFrame, label: int, session_id: int, step: int = STEP
) -> list[dict]:
    rows: list[dict] = []
    total = df["total_vibration"].values
    rms_x = df["rms_x"].values
    rms_y = df["rms_y"].values
    rms_z = df["rms_z"].values
    for start in range(0, len(df) - WINDOW_SIZE + 1, step):
        sl = slice(start, start + WINDOW_SIZE)
        feats = extract_features(total[sl], rms_x[sl], rms_y[sl], rms_z[sl])
        feats["label"] = label
        feats["session_id"] = session_id
        rows.append(feats)
    return rows


def build_dataset(base: Path, class_labels: tuple[str, ...], binary: bool) -> pd.DataFrame:
    label_to_idx = {name: i for i, name in enumerate(class_labels)}
    all_rows: list[dict] = []
    counts: dict[str, int] = {n: 0 for n in class_labels}
    next_session_id = 0

    normal_dir = base / "Normal_mod"
    if normal_dir.is_dir():
        for csv_path in sorted(normal_dir.rglob("vibration_log.csv")):
            df = load_session(csv_path)
            if df is None:
                continue
            rows = extract_session_features(
                df, label=label_to_idx["normal"], session_id=next_session_id
            )
            all_rows.extend(rows)
            counts["normal"] += 1
            next_session_id += 1

    deformed_dir = base / "Deformed_mod"
    if deformed_dir.is_dir():
        positions_on_disk = ("front_left", "front_right", "rear_left", "rear_right")
        for position in positions_on_disk:
            pos_dir = deformed_dir / position
            if not pos_dir.is_dir():
                continue
            target_label = "fault" if binary else position
            target_idx = label_to_idx[target_label]
            for csv_path in sorted(pos_dir.rglob("vibration_log.csv")):
                df = load_session(csv_path)
                if df is None:
                    continue
                rows = extract_session_features(
                    df, label=target_idx, session_id=next_session_id
                )
                all_rows.extend(rows)
                counts[target_label] += 1
                next_session_id += 1

    if not all_rows:
        print(f"Нет данных для обучения в {base}", file=sys.stderr)
        sys.exit(1)

    print("Сессий по классам:")
    for name in class_labels:
        print(f"  {name:12s} {counts[name]}")

    return pd.DataFrame(all_rows)


def make_classifier(kind: str, max_depth: int, min_samples_leaf: int):
    if kind == "tree":
        return DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=42,
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=max(1, min_samples_leaf - 2),
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    raise SystemExit(f"Неизвестный тип модели: {kind}")


def evaluate_groupwise(clf, X, y, groups, n_splits: int = 5) -> tuple[float, float, np.ndarray]:
    gkf = GroupKFold(n_splits=n_splits)
    accs = cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring="accuracy", n_jobs=-1)
    f1s = cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring="f1_macro", n_jobs=-1)
    return float(accs.mean()), float(f1s.mean()), accs


def main() -> None:
    here = _PROJECT_ROOT
    parser = argparse.ArgumentParser(
        description="Обучение детектора (multi-class: normal + 4 позиции винта)"
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=here / "datasets" / "set_03",
        help="Каталог с Normal_mod/ и Deformed_mod/<position>/",
    )
    parser.add_argument(
        "--model",
        choices=("tree", "rf"),
        default="rf",
        help="Тип модели: tree (одно дерево) или rf (Random Forest, по умолчанию)",
    )
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.95,
        help="Минимально приемлемая GroupKFold accuracy",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/propeller_fault_model.pkl",
        help="Путь к файлу модели относительно корня проекта",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Бинарный режим: 2 класса (normal vs fault), позиция игнорируется",
    )
    args = parser.parse_args()
    class_labels: tuple[str, ...] = (
        CLASS_LABELS_BINARY if args.binary else CLASS_LABELS_FULL
    )

    base = args.base.resolve()
    if not base.is_dir():
        print(f"База данных не найдена: {base}", file=sys.stderr)
        sys.exit(1)

    print(f"Источник: {base}")
    print(f"Режим: {'бинарный (normal/fault)' if args.binary else 'мультикласс (5 классов)'}")
    print("Сбор признаков (с FFT + корреляциями) …")
    dataset = build_dataset(base, class_labels, binary=args.binary)
    feature_cols = [c for c in dataset.columns if c not in ("label", "session_id")]
    X = dataset[feature_cols].values
    y = dataset["label"].values
    groups = dataset["session_id"].values

    print("\nОкон по классам:")
    for i, name in enumerate(class_labels):
        n = int(np.sum(y == i))
        print(f"  {name:12s} {n}")
    print(f"  всего окон:  {len(y)}")
    print(f"  всего сессий: {dataset['session_id'].nunique()}")

    clf = make_classifier(args.model, args.max_depth, args.min_samples_leaf)

    print("\n=== Честная валидация (GroupKFold по сессиям, без утечки) ===")
    acc_g, f1_g, acc_g_all = evaluate_groupwise(clf, X, y, groups, n_splits=5)
    print(f"GroupKFold accuracy : {acc_g:.4f}  (по фолдам: {np.round(acc_g_all,4).tolist()})")
    print(f"GroupKFold f1_macro : {f1_g:.4f}")

    print("\n=== Старая валидация (StratifiedKFold по окнам, для сравнения) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_s = float(cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1).mean())
    f1_s = float(cross_val_score(clf, X, y, cv=skf, scoring="f1_macro", n_jobs=-1).mean())
    print(f"StratifiedKFold accuracy : {acc_s:.4f}")
    print(f"StratifiedKFold f1_macro : {f1_s:.4f}")

    if acc_g < args.target_accuracy:
        print(
            f"\n[!] Честная accuracy {acc_g:.4f} ниже целевой {args.target_accuracy:.2f}.\n"
            f"    Это реалистичная оценка: дерево/лес не различает позицию идеально.\n"
            f"    Попробуйте --model rf, увеличить --max-depth, добавить данных.",
            file=sys.stderr,
        )

    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("\n=== Отчёт на полной выборке (для контроля) ===")
    print(classification_report(y, y_pred, target_names=list(class_labels)))
    print("Confusion matrix (rows=truth, cols=pred):")
    print("    " + " ".join(f"{lbl:>11s}" for lbl in class_labels))
    cm = confusion_matrix(y, y_pred, labels=list(range(len(class_labels))))
    for lbl, row in zip(class_labels, cm):
        print(f"{lbl:12s} " + " ".join(f"{v:>11d}" for v in row))

    if not args.binary:
        print("\n=== Confusion matrix по группам мотора (A/B/normal) ===")
        group_map = {i: PROP_GROUP[lbl] for i, lbl in enumerate(class_labels)}
        pairs = defaultdict(int)
        for t, p in zip(y, y_pred):
            pairs[(group_map[t], group_map[p])] += 1
        cats = ["A", "B", "-"]
        print("    " + "  ".join(f"{c:>4s}" for c in cats))
        for r in cats:
            print(f"{r:4s}" + "  ".join(f"{pairs[(r,c)]:>4d}" for c in cats))

    if args.model == "tree":
        print(f"\nГлубина: {clf.get_depth()}, листьев: {clf.get_n_leaves()}")
        print("\nДерево (top уровни):")
        print(export_text(clf, feature_names=feature_cols, max_depth=4))

    importances = (
        clf.feature_importances_
        if hasattr(clf, "feature_importances_")
        else np.zeros(len(feature_cols))
    )
    top_idx = np.argsort(importances)[::-1][:15]
    print("Топ-15 признаков:")
    for i in top_idx:
        print(f"  {feature_cols[i]:32s} {importances[i]:.4f}")

    model_path = (here / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "classifier": clf,
                "feature_names": feature_cols,
                "class_labels": list(class_labels),
                "position_ru": POSITION_RU,
                "prop_group": PROP_GROUP,
                "window_size": WINDOW_SIZE,
                "target_accuracy": args.target_accuracy,
                "cv_accuracy": acc_g,
                "cv_accuracy_strat": acc_s,
                "model_kind": args.model,
                "task": "binary" if args.binary else "multiclass",
            },
            f,
        )
    print(f"\nМодель сохранена: {model_path}")
    print(
        f"GroupKFold accuracy: {acc_g:.4f}  |  StratifiedKFold accuracy: {acc_s:.4f}"
        f"  |  target: {args.target_accuracy:.2f}"
    )


if __name__ == "__main__":
    main()
