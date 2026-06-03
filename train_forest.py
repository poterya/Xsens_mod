"""Обучение случайного леса на задаче «норма / поломка».

Подбор гиперпараметров (max_depth, min_samples_leaf) по F1 на val
для класса «поломка», построение кривой по числу деревьев, ROC-AUC.
Сохраняет графики в images/forest/ и модель в models/.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)

from features import build_all

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.prop_cycle": plt.cycler(color=["black", "0.35", "0.6"]),
    "image.cmap": "gray",
})

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "images" / "forest"
IMG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

N_TREES_CURVE = [10, 25, 50, 100, 200, 300, 500]
DEPTHS = [None, 5, 10, 15, 20]
LEAF_VALUES = [1, 2, 5, 10, 20]
RANDOM_STATE = 42
CLASS_NAMES = ["норма", "поломка"]


def _fit_rf(X, y, n_estimators=300, max_depth=None, min_samples_leaf=1):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(X, y)
    return clf


def plot_n_estimators_curve(scores_train, scores_val, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    ax.plot(N_TREES_CURVE, scores_train, color="black", linestyle="--", marker="o", label="F1 на обучении")
    ax.plot(N_TREES_CURVE, scores_val, color="black", linestyle="-", marker="s", label="F1 на валидации")
    ax.set_xlabel("Число деревьев в лесу")
    ax.set_ylabel("F1 (класс «поломка»)")
    ax.set_title("Зависимость качества от числа деревьев")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_validation_curves(curve_depth, curve_leaf, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    depths_show = [str(d) if d is not None else "None" for d in DEPTHS]
    axes[0].plot(depths_show, curve_depth["train"], color="black", linestyle="--", marker="o", label="train")
    axes[0].plot(depths_show, curve_depth["val"], color="black", linestyle="-", marker="s", label="val")
    axes[0].set_xlabel("max_depth")
    axes[0].set_ylabel("F1 (класс «поломка»)")
    axes[0].set_title("Кривая валидации: max_depth")
    axes[0].legend(loc="lower right")

    axes[1].plot([str(v) for v in LEAF_VALUES], curve_leaf["train"], color="black", linestyle="--", marker="o", label="train")
    axes[1].plot([str(v) for v in LEAF_VALUES], curve_leaf["val"], color="black", linestyle="-", marker="s", label="val")
    axes[1].set_xlabel("min_samples_leaf")
    axes[1].set_ylabel("F1 (класс «поломка»)")
    axes[1].set_title("Кривая валидации: min_samples_leaf")
    axes[1].legend(loc="lower right")

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
    ax.set_title("Топ важности признаков (случайный лес)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true, y_pred, path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=160)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Greys", colorbar=False, values_format="d"
    )
    ax.set_title("Матрица ошибок на тестовой выборке")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, proba, path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_value = float(auc(fpr, tpr))
    fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=160)
    ax.plot(fpr, tpr, color="black", linewidth=1.6, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("Доля ложно-положительных")
    ax.set_ylabel("Доля верно-положительных")
    ax.set_title("ROC-кривая на тестовой выборке")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return auc_value


def main() -> None:
    data = build_all(HERE / "dataset")
    Xtr, ytr, fnames = data["train"].X, data["train"].y, data["train"].feature_names
    Xva, yva = data["val"].X, data["val"].y
    Xte, yte = data["test"].X, data["test"].y

    curve_train, curve_val = [], []
    for n in N_TREES_CURVE:
        clf = _fit_rf(Xtr, ytr, n_estimators=n)
        curve_train.append(f1_score(ytr, clf.predict(Xtr)))
        curve_val.append(f1_score(yva, clf.predict(Xva)))
    plot_n_estimators_curve(curve_train, curve_val, IMG_DIR / "число_деревьев.png")
    best_n = N_TREES_CURVE[int(np.argmax(curve_val))]

    curve_depth = {"train": [], "val": []}
    for d in DEPTHS:
        clf = _fit_rf(Xtr, ytr, n_estimators=best_n, max_depth=d)
        curve_depth["train"].append(f1_score(ytr, clf.predict(Xtr)))
        curve_depth["val"].append(f1_score(yva, clf.predict(Xva)))
    best_depth = DEPTHS[int(np.argmax(curve_depth["val"]))]

    curve_leaf = {"train": [], "val": []}
    for v in LEAF_VALUES:
        clf = _fit_rf(Xtr, ytr, n_estimators=best_n, max_depth=best_depth, min_samples_leaf=v)
        curve_leaf["train"].append(f1_score(ytr, clf.predict(Xtr)))
        curve_leaf["val"].append(f1_score(yva, clf.predict(Xva)))
    best_leaf = LEAF_VALUES[int(np.argmax(curve_leaf["val"]))]

    plot_validation_curves(curve_depth, curve_leaf, IMG_DIR / "кривые_валидации.png")

    X_full = np.vstack([Xtr, Xva])
    y_full = np.concatenate([ytr, yva])
    final_clf = _fit_rf(
        X_full, y_full,
        n_estimators=best_n,
        max_depth=best_depth,
        min_samples_leaf=best_leaf,
    )

    plot_feature_importance(final_clf, fnames, IMG_DIR / "важность_признаков.png")
    test_pred = final_clf.predict(Xte)
    plot_confusion(yte, test_pred, IMG_DIR / "матрица_ошибок.png")
    proba = final_clf.predict_proba(Xte)[:, 1]
    auc_value = plot_roc(yte, proba, IMG_DIR / "roc_кривая.png")

    test_f1 = f1_score(yte, test_pred)
    report = classification_report(yte, test_pred, target_names=CLASS_NAMES, digits=3)

    print(f"Лучшее число деревьев по val: {best_n}")
    print(f"Лучшая глубина по val: {best_depth}")
    print(f"Лучший min_samples_leaf по val: {best_leaf}")
    print(f"F1 на тесте: {test_f1:.3f}")
    print(f"AUC на тесте: {auc_value:.3f}")
    print("Отчёт на тесте:\n" + report)

    with open(MODELS_DIR / "method_forest.pkl", "wb") as f:
        pickle.dump(
            {
                "model": final_clf,
                "feature_names": fnames,
                "best_n_estimators": best_n,
                "best_max_depth": best_depth,
                "best_min_samples_leaf": best_leaf,
                "test_f1": test_f1,
                "test_auc": auc_value,
                "classification_report": report,
                "n_trees_curve": {"x": N_TREES_CURVE, "train": curve_train, "val": curve_val},
                "depth_curve": {"x": [str(d) for d in DEPTHS], **curve_depth},
                "leaf_curve": {"x": LEAF_VALUES, **curve_leaf},
            },
            f,
        )


if __name__ == "__main__":
    main()
