"""Обучение дерева решений на задаче «норма / поломка».

Пайплайн: train -> подбор глубины по F1 на val -> переобучение train+val ->
проверка на test. Сохраняет графики в images/tree/ и модель в models/.
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
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

from features import LABEL_NAME, build_all

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.prop_cycle": plt.cycler(color=["black", "0.35", "0.6"]),
    "image.cmap": "gray",
})

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "images" / "tree"
IMG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEPTHS = list(range(2, 21))
RANDOM_STATE = 42
CLASS_NAMES = ["норма", "поломка"]


def fit_tree(X, y, depth):
    clf = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    clf.fit(X, y)
    return clf


def plot_depth_curve(train_scores, val_scores, best_depth, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    ax.plot(DEPTHS, train_scores, color="black", linestyle="--", marker="o", label="F1 на обучении")
    ax.plot(DEPTHS, val_scores, color="black", linestyle="-", marker="s", label="F1 на валидации")
    ax.axvline(best_depth, color="0.4", linestyle=":", linewidth=1.2, label=f"лучшая глубина = {best_depth}")
    ax.set_xlabel("Максимальная глубина дерева")
    ax.set_ylabel("F1 (класс «поломка»)")
    ax.set_title("Зависимость качества от глубины дерева")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(clf, feature_names, path: Path, top_k: int = 15):
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1][:top_k]
    names = [feature_names[i] for i in idx][::-1]
    values = imp[idx][::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=160)
    ax.barh(names, values, color="0.25", edgecolor="black")
    ax.set_xlabel("Важность признака")
    ax.set_title("Топ важности признаков (дерево решений)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_tree_structure(clf, feature_names, path: Path):
    fig, ax = plt.subplots(figsize=(18, 10), dpi=160)
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        filled=False,
        impurity=False,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Структура обученного дерева решений")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true, y_pred, path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=160)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Greys", colorbar=False, values_format="d")
    ax.set_title("Матрица ошибок на тестовой выборке")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _decision_path_text(clf, x_row, feature_names) -> list[str]:
    tree_ = clf.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    node = 0
    lines: list[str] = []
    while tree_.children_left[node] != tree_.children_right[node]:
        f = feature[node]
        t = threshold[node]
        v = float(x_row[f])
        if v <= t:
            lines.append(f"{feature_names[f]} = {v:.4f} ≤ {t:.4f}  →  влево")
            node = tree_.children_left[node]
        else:
            lines.append(f"{feature_names[f]} = {v:.4f} > {t:.4f}  →  вправо")
            node = tree_.children_right[node]
    values = tree_.value[node][0]
    pred = int(np.argmax(values))
    lines.append(f"лист: {CLASS_NAMES[pred]} (классы: норма={int(values[0])}, поломка={int(values[1])})")
    return lines


def plot_decision_paths(clf, X, y, feature_names, path: Path):
    idx_norm = int(np.where(y == 0)[0][0])
    idx_fault = int(np.where(y == 1)[0][0])
    lines_norm = _decision_path_text(clf, X[idx_norm], feature_names)
    lines_fault = _decision_path_text(clf, X[idx_fault], feature_names)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=160)
    for ax, title, lines in (
        (axes[0], "Путь решения: норма", lines_norm),
        (axes[1], "Путь решения: поломка", lines_fault),
    ):
        ax.axis("off")
        ax.set_title(title, fontsize=12)
        for i, ln in enumerate(lines):
            ax.text(0.02, 0.95 - i * 0.07, ln, fontsize=10, family="monospace",
                    transform=ax.transAxes, color="black")
    fig.suptitle("Прохождение примеров от корня к листу", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = build_all(HERE / "dataset")
    Xtr, ytr, ftr = data["train"].X, data["train"].y, data["train"].feature_names
    Xva, yva = data["val"].X, data["val"].y
    Xte, yte = data["test"].X, data["test"].y

    train_scores: list[float] = []
    val_scores: list[float] = []
    for d in DEPTHS:
        clf_d = fit_tree(Xtr, ytr, d)
        train_scores.append(f1_score(ytr, clf_d.predict(Xtr)))
        val_scores.append(f1_score(yva, clf_d.predict(Xva)))
    best_idx = int(np.argmax(val_scores))
    best_depth = DEPTHS[best_idx]

    plot_depth_curve(train_scores, val_scores, best_depth,
                     IMG_DIR / "глубина_качество.png")

    X_full = np.vstack([Xtr, Xva])
    y_full = np.concatenate([ytr, yva])
    final_clf = fit_tree(X_full, y_full, best_depth)

    plot_tree_structure(final_clf, ftr, IMG_DIR / "структура_дерева.png")
    plot_feature_importance(final_clf, ftr, IMG_DIR / "важность_признаков.png")
    plot_decision_paths(final_clf, Xte, yte, ftr, IMG_DIR / "путь_решения.png")
    plot_confusion(yte, final_clf.predict(Xte), IMG_DIR / "матрица_ошибок.png")

    test_pred = final_clf.predict(Xte)
    test_f1 = f1_score(yte, test_pred)
    report = classification_report(
        yte, test_pred, target_names=CLASS_NAMES, digits=3
    )

    print(f"Лучшая глубина по val: {best_depth}")
    print(f"F1 на тесте: {test_f1:.3f}")
    print("Отчёт на тесте:\n" + report)

    with open(MODELS_DIR / "method_tree.pkl", "wb") as f:
        pickle.dump(
            {
                "model": final_clf,
                "feature_names": ftr,
                "best_depth": best_depth,
                "train_f1_curve": train_scores,
                "val_f1_curve": val_scores,
                "test_f1": test_f1,
                "classification_report": report,
            },
            f,
        )


if __name__ == "__main__":
    main()
