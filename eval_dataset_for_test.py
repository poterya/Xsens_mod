"""Оценка обученных моделей на независимой выборке dataset_for_test.

Загружает models/method_{tree,forest,mlp}.pkl, формирует окна признаков
из dataset_for_test через features.build_split, считает метрики и
строит ч/б графики: матрица ошибок и ROC по каждой модели,
сравнительный столбчатый график по всем моделям и сводную таблицу.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from features import build_split

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.prop_cycle": plt.cycler(color=["black", "0.35", "0.6"]),
    "image.cmap": "gray",
})

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset_for_test"
MODELS_DIR = HERE / "models"
IMG_DIR = HERE / "images" / "dataset_for_test"
DOCS_DIR = HERE / "docs"
CLASS_NAMES = ["норма", "поломка"]

MODELS = (
    ("tree",   "Дерево решений"),
    ("forest", "Случайный лес"),
    ("mlp",    "MLP"),
)


def _load_model(name: str):
    with open(MODELS_DIR / f"method_{name}.pkl", "rb") as f:
        return pickle.load(f)


def _plot_confusion(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=160)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Greys", colorbar=False, values_format="d"
    )
    ax.set_title(title)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(y_true, proba, title: str, path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_value = float(auc(fpr, tpr))
    fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=160)
    ax.plot(fpr, tpr, color="black", linewidth=1.6, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("Доля ложно-положительных")
    ax.set_ylabel("Доля верно-положительных")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return auc_value


def _plot_comparison(rows: list[dict], path: Path) -> None:
    metrics = [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"),
               ("accuracy", "Accuracy"), ("auc", "AUC")]
    x = np.arange(len(metrics))
    width = 0.27
    styles = [
        dict(color="0.20", hatch=""),
        dict(color="0.55", hatch="//"),
        dict(color="0.85", hatch="xx"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=160)
    for i, r in enumerate(rows):
        vals = [r[k] for k, _ in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width=width,
                      label=r["label"], edgecolor="black",
                      color=styles[i]["color"], hatch=styles[i]["hatch"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение моделей на dataset_for_test")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not DATASET.is_dir():
        raise SystemExit(f"Нет каталога {DATASET}")

    ws = build_split(DATASET.parent, DATASET.name)
    X, y = ws.X, ws.y
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    n_flights = len(set(ws.groups.tolist()))
    print(f"dataset_for_test: окон={len(y)}, норма={n_neg}, поломка={n_pos}, полётов={n_flights}")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for key, label in MODELS:
        out_dir = IMG_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle = _load_model(key)
        model = bundle["model"]
        y_pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

        f1 = float(f1_score(y, y_pred))
        prec = float(precision_score(y, y_pred, zero_division=0))
        rec = float(recall_score(y, y_pred, zero_division=0))
        acc = float(np.mean(y_pred == y))
        _plot_confusion(y, y_pred, f"{label}: матрица ошибок", out_dir / "матрица_ошибок.png")
        auc_value = _plot_roc(y, proba, f"{label}: ROC", out_dir / "roc_кривая.png") if proba is not None else float("nan")
        report = classification_report(y, y_pred, target_names=CLASS_NAMES, digits=3)

        print(f"\n=== {label} ===")
        print(f"F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}  Acc={acc:.3f}  AUC={auc_value:.3f}")
        print(report)

        rows.append({
            "key": key, "label": label,
            "f1": f1, "precision": prec, "recall": rec, "accuracy": acc, "auc": auc_value,
            "report": report,
        })

    _plot_comparison(rows, IMG_DIR / "сравнение_моделей.png")

    md = []
    md.append("# Оценка моделей на независимой выборке dataset_for_test\n\n")
    md.append("Выборка собрана из полётов, которые не использовались ни в обучении, "
              "ни в валидации, ни в тесте: 10 деформированных полётов из исключённых при "
              "балансировке + нормальные полёты, добавленные отдельно. У нормальных полётов "
              "обрезаны по 20 % с каждого края (взлёт/посадка).\n\n")
    md.append(f"## Состав выборки\n\n- окон всего: {len(y)};\n- норма: {n_neg};\n- поломка: {n_pos};\n"
              f"- уникальных полётов: {n_flights}.\n\n")
    md.append("## Сводная таблица\n\n")
    md.append("| Модель | F1 (поломка) | Precision | Recall | Accuracy | AUC |\n")
    md.append("|--------|--------------|-----------|--------|----------|-----|\n")
    for r in rows:
        md.append(f"| {r['label']} | {r['f1']:.3f} | {r['precision']:.3f} | "
                  f"{r['recall']:.3f} | {r['accuracy']:.3f} | {r['auc']:.3f} |\n")
    md.append("\n## Сравнительный график\n\n")
    md.append("![сравнение моделей](../images/dataset_for_test/сравнение_моделей.png)\n\n")
    md.append("## Детальные отчёты\n")
    for r in rows:
        md.append(f"\n### {r['label']}\n\n")
        md.append("```\n" + r["report"] + "\n```\n\n")
        md.append(f"![матрица ошибок](../images/dataset_for_test/{r['key']}/матрица_ошибок.png)\n\n")
        md.append(f"![ROC](../images/dataset_for_test/{r['key']}/roc_кривая.png)\n")

    out_md = DOCS_DIR / "dataset_for_test_results.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print("\nсохранено:", out_md)


if __name__ == "__main__":
    main()
