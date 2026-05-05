#!/usr/bin/env python3
"""Train v2.0 neural detectors on datasets/set_05.

For each task (binary, position) two independent neural variants are trained:
  - "forest": small bagging ensemble of MLPs (MLPForest);
  - "tree":   single compact MLP.

The dataset must already be split as:
  datasets/set_05/train/...
  datasets/set_05/val/...
  datasets/set_05/test/...

Hyperparameters are selected by val accuracy/F1 inside each variant; the
selected model is reported on test and saved as a pickle bundle.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_extraction import extract_features
from nn_forest import MLPForest

WINDOW_SIZE = 50
STEP = 10

POSITION_LABELS: tuple[str, ...] = (
    "normal",
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
)
BINARY_LABELS: tuple[str, ...] = ("normal", "fault")


@dataclass(frozen=True)
class DatasetPart:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]


def _load_session(csv_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    cols = ["total_vibration", "rms_x", "rms_y", "rms_z"]
    if not set(cols).issubset(df.columns):
        return None
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=cols, inplace=True)
    if len(df) < WINDOW_SIZE:
        return None
    return df


def _csvs_for_label(split_root: Path, label: str) -> list[Path]:
    if label == "normal":
        root = split_root / "Normal_mod"
    else:
        root = split_root / "Deformed_mod" / label
    if not root.is_dir():
        return []
    return sorted(root.rglob("vibration_log.csv"))


def _extract_rows(df: pd.DataFrame, label_idx: int) -> list[dict]:
    rows: list[dict] = []
    total = df["total_vibration"].to_numpy(dtype=float)
    rms_x = df["rms_x"].to_numpy(dtype=float)
    rms_y = df["rms_y"].to_numpy(dtype=float)
    rms_z = df["rms_z"].to_numpy(dtype=float)
    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP):
        sl = slice(start, start + WINDOW_SIZE)
        feats = extract_features(total[sl], rms_x[sl], rms_y[sl], rms_z[sl])
        feats["label"] = label_idx
        rows.append(feats)
    return rows


def build_split(split_root: Path, task: str) -> DatasetPart:
    if task == "binary":
        class_labels = BINARY_LABELS
    elif task == "position":
        class_labels = POSITION_LABELS
    else:
        raise ValueError(f"Unknown task: {task}")

    label_to_idx = {label: i for i, label in enumerate(class_labels)}
    all_rows: list[dict] = []
    counts = {label: 0 for label in class_labels}

    for label in POSITION_LABELS:
        for csv_path in _csvs_for_label(split_root, label):
            df = _load_session(csv_path)
            if df is None:
                continue
            target_label = "fault" if task == "binary" and label != "normal" else label
            if target_label not in label_to_idx:
                continue
            rows = _extract_rows(df, label_to_idx[target_label])
            if rows:
                all_rows.extend(rows)
                counts[target_label] += 1

    if not all_rows:
        raise SystemExit(f"Нет данных для {task} в {split_root}")

    frame = pd.DataFrame(all_rows)
    feature_names = [c for c in frame.columns if c != "label"]
    print(f"\n{task} / {split_root.name}:")
    for label in class_labels:
        print(f"  sessions {label:12s} {counts[label]}")
    for i, label in enumerate(class_labels):
        print(f"  windows  {label:12s} {int(np.sum(frame['label'].to_numpy() == i))}")
    print(f"  total windows: {len(frame)}")

    return DatasetPart(
        X=frame[feature_names].to_numpy(dtype=float),
        y=frame["label"].to_numpy(dtype=int),
        feature_names=feature_names,
    )


def _candidate_configs(variant: str, task: str) -> list[dict]:
    if variant == "forest":
        return [
            {"hidden_layer_sizes": (96, 48), "alpha": 1e-4, "n_estimators": 5},
            {"hidden_layer_sizes": (128, 64), "alpha": 3e-4, "n_estimators": 5},
        ]
    if variant == "tree":
        if task == "binary":
            return [
                {"hidden_layer_sizes": (32,), "alpha": 1e-4},
                {"hidden_layer_sizes": (48, 24), "alpha": 3e-4},
            ]
        return [
            {"hidden_layer_sizes": (48, 24), "alpha": 1e-4},
            {"hidden_layer_sizes": (64, 32), "alpha": 3e-4},
        ]
    raise ValueError(f"Unknown variant: {variant}")


def _make_model(variant: str, params: dict, seed: int) -> Pipeline:
    if variant == "forest":
        clf = MLPForest(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            alpha=params["alpha"],
            n_estimators=int(params.get("n_estimators", 5)),
            max_samples_frac=0.8,
            max_features_frac=1.0,
            max_iter=200,
            random_state=seed,
        )
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            activation="relu",
            solver="adam",
            alpha=params["alpha"],
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=False,
            random_state=seed,
        )
    return Pipeline([("scaler", StandardScaler()), ("nn", clf)])


def _score(model: Pipeline, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pred = model.predict(X)
    return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro"))


def _plot_loss(out_dir: Path, name: str, history: dict[str, list[list[float]]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, curves in history.items():
        for i, curve in enumerate(curves):
            line_label = label if i == 0 else None
            ax.plot(curve, linewidth=1.0, alpha=0.85, label=line_label)
    ax.set_title(f"NN v2.0 loss curves ({name})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8)
    fig.tight_layout()
    path = out_dir / f"nn_v2_{name}_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"График loss: {path}")


def _plot_metric_bars(out_dir: Path, name: str, rows: list[dict]) -> None:
    names = [r["name"] for r in rows]
    val_acc = [r["val_accuracy"] for r in rows]
    val_f1 = [r["val_f1_macro"] for r in rows]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, val_acc, width, label="val accuracy")
    ax.bar(x + width / 2, val_f1, width, label="val f1_macro")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"NN v2.0 validation metrics ({name})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"nn_v2_{name}_val_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"График val metrics: {path}")


def _plot_confusion(
    out_dir: Path,
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: tuple[str, ...],
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_labels))
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"NN v2.0 test confusion matrix ({name})")
    fig.tight_layout()
    path = out_dir / f"nn_v2_{name}_test_confusion.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix: {path}")


def _loss_curves_from_pipeline(model: Pipeline) -> list[list[float]]:
    nn = model.named_steps["nn"]
    if isinstance(nn, MLPForest):
        return nn.loss_curves_
    return [list(getattr(nn, "loss_curve_", []))]


def train_variant(
    task: str,
    variant: str,
    train: DatasetPart,
    val: DatasetPart,
    test: DatasetPart,
    plots_dir: Path,
    models_dir: Path,
    seed: int,
) -> dict:
    class_labels = BINARY_LABELS if task == "binary" else POSITION_LABELS
    name = f"{variant}_{task}"
    history: dict[str, list[list[float]]] = {}
    results: list[dict] = []
    best_model: Pipeline | None = None
    best_row: dict | None = None

    for idx, params in enumerate(_candidate_configs(variant, task), 1):
        cfg_name = f"{variant}{idx}_{params['hidden_layer_sizes']}_a{params['alpha']}"
        print(f"\n=== {name}: train {cfg_name} ===")
        model = _make_model(variant, params, seed + idx)
        model.fit(train.X, train.y)
        train_acc, train_f1 = _score(model, train.X, train.y)
        val_acc, val_f1 = _score(model, val.X, val.y)
        history[cfg_name] = _loss_curves_from_pipeline(model)
        row = {
            "name": cfg_name,
            "params": params,
            "train_accuracy": train_acc,
            "train_f1_macro": train_f1,
            "val_accuracy": val_acc,
            "val_f1_macro": val_f1,
        }
        results.append(row)
        print(
            f"train acc={train_acc:.4f} f1={train_f1:.4f} | "
            f"val acc={val_acc:.4f} f1={val_f1:.4f}"
        )
        if best_row is None or (val_f1, val_acc) > (
            best_row["val_f1_macro"],
            best_row["val_accuracy"],
        ):
            best_row = row
            best_model = model

    assert best_model is not None
    assert best_row is not None

    y_pred = best_model.predict(test.X)
    test_acc = float(accuracy_score(test.y, y_pred))
    test_f1 = float(f1_score(test.y, y_pred, average="macro"))
    print(f"\n=== {name}: TEST для лучшей модели ===")
    print(f"best: {best_row['name']}")
    print(f"test accuracy={test_acc:.4f} | test f1_macro={test_f1:.4f}")
    print(classification_report(test.y, y_pred, target_names=list(class_labels)))

    _plot_loss(plots_dir, name, history)
    _plot_metric_bars(plots_dir, name, results)
    _plot_confusion(plots_dir, name, test.y, y_pred, class_labels)

    model_path = models_dir / f"propeller_fault_nn_v2_{variant}_{task}_set05.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "classifier": best_model,
                "feature_names": train.feature_names,
                "class_labels": list(class_labels),
                "window_size": WINDOW_SIZE,
                "step": STEP,
                "task": task,
                "variant": variant,
                "model_kind": f"nn_v2_{variant}",
                "version": "2.0",
                "best_config": best_row,
                "candidates": results,
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
            },
            f,
        )
    print(f"Модель сохранена: {model_path}")
    return {
        "task": task,
        "variant": variant,
        "model_path": str(model_path),
        "best": best_row,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train neural v2.0 forest and tree models on set_05"
    )
    parser.add_argument("--set05", type=Path, default=here / "datasets" / "set_05")
    parser.add_argument(
        "--plots-dir", type=Path, default=here / "training_plots_v2",
        help="Папка для графиков обучения",
    )
    parser.add_argument("--models-dir", type=Path, default=here)
    parser.add_argument(
        "--task",
        choices=("all", "binary", "position"),
        default="all",
    )
    parser.add_argument(
        "--variant",
        choices=("all", "forest", "tree"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        if not (args.set05 / split).is_dir():
            raise SystemExit(f"Нет {split} split: {args.set05 / split}")

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    tasks = ("binary", "position") if args.task == "all" else (args.task,)
    variants = ("forest", "tree") if args.variant == "all" else (args.variant,)

    summary: list[dict] = []
    for task in tasks:
        train = build_split(args.set05 / "train", task)
        val = build_split(args.set05 / "val", task)
        test = build_split(args.set05 / "test", task)
        for variant in variants:
            summary.append(
                train_variant(
                    task,
                    variant,
                    train,
                    val,
                    test,
                    args.plots_dir.resolve(),
                    args.models_dir.resolve(),
                    args.seed,
                )
            )

    summary_path = args.plots_dir / "nn_v2_training_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
