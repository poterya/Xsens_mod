"""Оценка обученных моделей на независимой выборке dataset/test_2.

Загружает models/{tree,forest,mlp}.pkl, формирует окна признаков по
test_2 через features.build_split и считает метрики и матрицы ошибок.
Сохраняет результаты в images/test_2/{tree,forest,mlp}/ и docs/test_2_results.md.
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
DATASET = HERE / "dataset"
MODELS_DIR = HERE / "models"
IMG_DIR = HERE / "images" / "test_2"
DOCS_DIR = HERE / "docs"
CLASS_NAMES = ["норма", "поломка"]


def _plot_confusion(y_true, y_pred, title: str, path: Path):
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


def _plot_roc(y_true, proba, title: str, path: Path) -> float | None:
    if proba is None:
        return None
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


def _load_model(name: str):
    p = MODELS_DIR / f"{name}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def _predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            return None
    return None


def main() -> None:
    ws = build_split(DATASET, "test_2")
    X, y = ws.X, ws.y
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    print(f"test_2: окон {len(y)}, норма {n_neg}, поломка {n_pos}")
    print(f"уникальных полётов: {len(set(ws.groups.tolist()))}")

    models = {
        "tree":   ("Дерево решений", _load_model("tree")),
        "forest": ("Случайный лес",  _load_model("forest")),
        "mlp":    ("MLP",            _load_model("mlp")),
    }

    rows = []
    for key, (label, bundle) in models.items():
        if bundle is None:
            print(f"[пропуск] {key}: модель не найдена")
            continue
        out_dir = IMG_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)
        model = bundle["model"]
        y_pred = model.predict(X)
        proba = _predict_proba(model, X)
        f1 = f1_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        acc = float(np.mean(y_pred == y))
        _plot_confusion(y, y_pred, f"{label}: матрица ошибок (test_2)", out_dir / "матрица_ошибок.png")
        auc_value = _plot_roc(y, proba, f"{label}: ROC (test_2)", out_dir / "roc_кривая.png") if proba is not None else None
        report = classification_report(y, y_pred, target_names=CLASS_NAMES, digits=3)
        print(f"\n=== {label} ===")
        print(f"F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}  Acc={acc:.3f}  AUC={auc_value if auc_value is None else round(auc_value,3)}")
        print(report)
        rows.append({
            "key": key,
            "label": label,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "accuracy": acc,
            "auc": auc_value,
            "report": report,
        })

    md_lines = []
    md_lines.append("# Результаты на независимой выборке test_2\n")
    md_lines.append("Назначение выборки: проверка обобщающей способности моделей "
                    "и подозрения на утечку данных. test_2 состоит из полётов "
                    "новой даты (5 мая 2026), которых нет ни в train, ни в val, ни в test.\n")
    md_lines.append("## Состав выборки\n")
    md_lines.append(f"- окон всего: {len(y)};\n- норма: {n_neg};\n- поломка: {n_pos};\n"
                    f"- уникальных полётов: {len(set(ws.groups.tolist()))}.\n")
    md_lines.append("## Сводная таблица\n")
    md_lines.append("| Модель | F1 (поломка) | Precision | Recall | Accuracy | AUC |\n"
                    "|--------|--------------|-----------|--------|----------|-----|\n")
    for r in rows:
        auc_s = "—" if r["auc"] is None else f"{r['auc']:.3f}"
        md_lines.append(f"| {r['label']} | {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['accuracy']:.3f} | {auc_s} |\n")
    md_lines.append("\n## Детальные отчёты\n")
    for r in rows:
        md_lines.append(f"### {r['label']}\n")
        md_lines.append("```\n" + r["report"] + "\n```\n")
        md_lines.append(f"![матрица ошибок](../images/test_2/{r['key']}/матрица_ошибок.png)\n")
        if r["auc"] is not None:
            md_lines.append(f"![ROC](../images/test_2/{r['key']}/roc_кривая.png)\n")
        md_lines.append("\n")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "test_2_results.md").write_text("".join(md_lines), encoding="utf-8")
    print("\nсохранено:", DOCS_DIR / "test_2_results.md")


if __name__ == "__main__":
    main()
