"""Обучение многослойного перцептрона (MLP) на задаче «норма / поломка».

Pipeline = StandardScaler + MLPClassifier. Подбор гиперпараметров на val,
финальное переобучение на train+val, оценка на test. Сохраняет графики
в images/mlp/ и модель в models/.
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
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import build_all

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.prop_cycle": plt.cycler(color=["black", "0.35", "0.6"]),
    "image.cmap": "gray",
})

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "images" / "mlp"
IMG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HIDDEN_VARIANTS = [(16,), (32,), (64,), (32, 16), (64, 32), (64, 32, 16)]
ALPHA_VALUES = [1e-5, 1e-4, 1e-3, 1e-2]
LEARNING_RATE_INITS = [1e-4, 5e-4, 1e-3, 5e-3]
RANDOM_STATE = 42
MAX_ITER = 300
CLASS_NAMES = ["норма", "поломка"]


def _pipe(hidden, alpha=1e-4, lr=1e-3, max_iter=MAX_ITER, early=False):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size=128,
            learning_rate_init=lr,
            max_iter=max_iter,
            early_stopping=early,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        )),
    ])


def plot_loss_curve(loss, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    ax.plot(range(1, len(loss) + 1), loss, color="black", linewidth=1.5)
    ax.set_xlabel("Эпоха обучения")
    ax.set_ylabel("Значение функции потерь")
    ax.set_title("Кривая потерь MLP")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(train_sizes, train_f1, val_f1, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    ax.plot(train_sizes, train_f1, color="black", linestyle="--", marker="o", label="F1 на обучении")
    ax.plot(train_sizes, val_f1, color="black", linestyle="-", marker="s", label="F1 на валидации")
    ax.set_xlabel("Размер обучающей выборки")
    ax.set_ylabel("F1 (класс «поломка»)")
    ax.set_title("Кривые обучения MLP")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_hidden_curve(labels, train_scores, val_scores, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    xs = np.arange(len(labels))
    ax.plot(xs, train_scores, color="black", linestyle="--", marker="o", label="train")
    ax.plot(xs, val_scores, color="black", linestyle="-", marker="s", label="val")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_xlabel("Архитектура скрытых слоёв")
    ax.set_ylabel("F1 (класс «поломка»)")
    ax.set_title("Кривая валидации: размер скрытых слоёв")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_param_curve(values, train_scores, val_scores, xlabel, title, path: Path, log=True):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    ax.plot(values, train_scores, color="black", linestyle="--", marker="o", label="train")
    ax.plot(values, val_scores, color="black", linestyle="-", marker="s", label="val")
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("F1 (класс «поломка»)")
    ax.set_title(title)
    ax.legend(loc="lower right")
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


def plot_first_layer_weights(pipe: Pipeline, feature_names, path: Path):
    mlp: MLPClassifier = pipe.named_steps["mlp"]
    W = mlp.coefs_[0]
    vmax = float(np.abs(W).max())
    fig, ax = plt.subplots(figsize=(max(8, 0.18 * W.shape[1] + 4), max(8, 0.18 * W.shape[0] + 2)), dpi=160)
    im = ax.imshow(W, cmap="gray", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_xlabel("Нейрон первого скрытого слоя")
    ax.set_ylabel("Признак входа")
    ax.set_title("Веса первого скрытого слоя MLP")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = build_all(HERE / "dataset")
    Xtr, ytr, fnames = data["train"].X, data["train"].y, data["train"].feature_names
    Xva, yva = data["val"].X, data["val"].y
    Xte, yte = data["test"].X, data["test"].y

    hidden_train, hidden_val = [], []
    for h in HIDDEN_VARIANTS:
        pipe = _pipe(hidden=h, max_iter=200)
        pipe.fit(Xtr, ytr)
        hidden_train.append(f1_score(ytr, pipe.predict(Xtr)))
        hidden_val.append(f1_score(yva, pipe.predict(Xva)))
    best_hidden = HIDDEN_VARIANTS[int(np.argmax(hidden_val))]
    plot_hidden_curve(
        [str(h) for h in HIDDEN_VARIANTS],
        hidden_train, hidden_val,
        IMG_DIR / "кривая_валидации_скрытые_слои.png",
    )

    alpha_train, alpha_val = [], []
    for a in ALPHA_VALUES:
        pipe = _pipe(hidden=best_hidden, alpha=a, max_iter=200)
        pipe.fit(Xtr, ytr)
        alpha_train.append(f1_score(ytr, pipe.predict(Xtr)))
        alpha_val.append(f1_score(yva, pipe.predict(Xva)))
    best_alpha = ALPHA_VALUES[int(np.argmax(alpha_val))]
    plot_param_curve(
        ALPHA_VALUES, alpha_train, alpha_val,
        xlabel="alpha (L2-регуляризация)",
        title="Кривая валидации: alpha",
        path=IMG_DIR / "кривая_валидации_alpha.png",
        log=True,
    )

    lr_train, lr_val = [], []
    for lr in LEARNING_RATE_INITS:
        pipe = _pipe(hidden=best_hidden, alpha=best_alpha, lr=lr, max_iter=200)
        pipe.fit(Xtr, ytr)
        lr_train.append(f1_score(ytr, pipe.predict(Xtr)))
        lr_val.append(f1_score(yva, pipe.predict(Xva)))
    best_lr = LEARNING_RATE_INITS[int(np.argmax(lr_val))]
    plot_param_curve(
        LEARNING_RATE_INITS, lr_train, lr_val,
        xlabel="learning_rate_init",
        title="Кривая валидации: learning_rate_init",
        path=IMG_DIR / "кривая_валидации_lr.png",
        log=True,
    )

    sizes_fraction = [0.1, 0.25, 0.5, 0.75, 1.0]
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(len(Xtr))
    lc_sizes, lc_train, lc_val = [], [], []
    for frac in sizes_fraction:
        n = int(len(Xtr) * frac)
        idx = perm[:n]
        pipe = _pipe(hidden=best_hidden, alpha=best_alpha, lr=best_lr, max_iter=200)
        pipe.fit(Xtr[idx], ytr[idx])
        lc_sizes.append(n)
        lc_train.append(f1_score(ytr[idx], pipe.predict(Xtr[idx])))
        lc_val.append(f1_score(yva, pipe.predict(Xva)))
    plot_learning_curves(lc_sizes, lc_train, lc_val, IMG_DIR / "кривые_обучения.png")

    X_full = np.vstack([Xtr, Xva])
    y_full = np.concatenate([ytr, yva])
    final_pipe = _pipe(
        hidden=best_hidden,
        alpha=best_alpha,
        lr=best_lr,
        max_iter=MAX_ITER,
        early=False,
    )
    final_pipe.fit(X_full, y_full)

    loss = final_pipe.named_steps["mlp"].loss_curve_
    plot_loss_curve(loss, IMG_DIR / "кривая_потерь.png")

    test_pred = final_pipe.predict(Xte)
    plot_confusion(yte, test_pred, IMG_DIR / "матрица_ошибок.png")
    proba = final_pipe.predict_proba(Xte)[:, 1]
    auc_value = plot_roc(yte, proba, IMG_DIR / "roc_кривая.png")

    plot_first_layer_weights(final_pipe, fnames, IMG_DIR / "веса_первого_слоя.png")

    test_f1 = f1_score(yte, test_pred)
    report = classification_report(yte, test_pred, target_names=CLASS_NAMES, digits=3)

    print(f"Лучшая архитектура: {best_hidden}")
    print(f"Лучшее alpha: {best_alpha}")
    print(f"Лучшее learning_rate_init: {best_lr}")
    print(f"Эпох обучения финальной модели: {len(loss)}")
    print(f"F1 на тесте: {test_f1:.3f}")
    print(f"AUC на тесте: {auc_value:.3f}")
    print("Отчёт на тесте:\n" + report)

    with open(MODELS_DIR / "method_mlp.pkl", "wb") as f:
        pickle.dump(
            {
                "model": final_pipe,
                "feature_names": fnames,
                "best_hidden": best_hidden,
                "best_alpha": best_alpha,
                "best_lr": best_lr,
                "loss_curve": loss,
                "hidden_curve": {"x": [str(h) for h in HIDDEN_VARIANTS], "train": hidden_train, "val": hidden_val},
                "alpha_curve": {"x": ALPHA_VALUES, "train": alpha_train, "val": alpha_val},
                "lr_curve": {"x": LEARNING_RATE_INITS, "train": lr_train, "val": lr_val},
                "learning_curve": {"sizes": lc_sizes, "train": lc_train, "val": lc_val},
                "test_f1": test_f1,
                "test_auc": auc_value,
                "classification_report": report,
            },
            f,
        )


if __name__ == "__main__":
    main()
