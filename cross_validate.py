"""GroupKFold-кросс-валидация по полёту для tree/forest/mlp.

Берёт гиперпараметры, подобранные на отложенной валидации (из
models/method_*.pkl), и прогоняет 5-fold GroupKFold по flight_id,
объединяя train+val+test в один пул. Сохраняет таблицу метрик
в docs/cross_validation.md и графики в images/cv/.
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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
IMG_DIR = HERE / "images" / "cv"
IMG_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = HERE / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42
CLASS_NAMES = ["норма", "поломка"]


def _load_meta(name: str) -> dict:
    with open(MODELS_DIR / f"method_{name}.pkl", "rb") as f:
        return pickle.load(f)


def _build_pool():
    parts = [build_split(DATASET, s) for s in ("train", "val", "test")]
    X = np.vstack([p.X for p in parts])
    y = np.concatenate([p.y for p in parts])
    groups = np.concatenate([p.groups for p in parts])
    feature_names = parts[0].feature_names
    return X, y, groups, feature_names


def _make_estimator(name: str, meta: dict):
    if name == "tree":
        return DecisionTreeClassifier(
            max_depth=meta["best_depth"],
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if name == "forest":
        return RandomForestClassifier(
            n_estimators=meta["best_n_estimators"],
            max_depth=meta["best_max_depth"],
            min_samples_leaf=meta["best_min_samples_leaf"],
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=meta["best_hidden"],
                activation="relu",
                solver="adam",
                alpha=meta["best_alpha"],
                batch_size=128,
                learning_rate_init=meta["best_lr"],
                max_iter=300,
                early_stopping=False,
                n_iter_no_change=15,
                random_state=RANDOM_STATE,
            )),
        ])
    raise ValueError(name)


def _proba(model, X) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            return None
    return None


def main() -> None:
    X, y, groups, _ = _build_pool()
    print(f"всего окон: {len(y)}, уникальных полётов: {len(set(groups.tolist()))}")
    print(f"норма={int((y == 0).sum())}, поломка={int((y == 1).sum())}")

    metas = {
        "tree":   ("Дерево решений", _load_meta("tree")),
        "forest": ("Случайный лес",  _load_meta("forest")),
        "mlp":    ("MLP",            _load_meta("mlp")),
    }

    cv = GroupKFold(n_splits=N_SPLITS)
    results: dict[str, dict] = {}

    for name, (label, meta) in metas.items():
        f1s, aucs, precs, recs, accs = [], [], [], [], []
        for k, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups=groups), 1):
            est = _make_estimator(name, meta)
            est.fit(X[tr_idx], y[tr_idx])
            y_pred = est.predict(X[te_idx])
            proba = _proba(est, X[te_idx])
            f1 = f1_score(y[te_idx], y_pred)
            prec = precision_score(y[te_idx], y_pred, zero_division=0)
            rec = recall_score(y[te_idx], y_pred, zero_division=0)
            acc = accuracy_score(y[te_idx], y_pred)
            auc = float(roc_auc_score(y[te_idx], proba)) if proba is not None else float("nan")
            n_train_flights = len(set(groups[tr_idx].tolist()))
            n_test_flights  = len(set(groups[te_idx].tolist()))
            print(f"[{name}] fold {k}: train_flights={n_train_flights}  test_flights={n_test_flights}  "
                  f"F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}  Acc={acc:.3f}  AUC={auc:.3f}")
            f1s.append(f1); aucs.append(auc); precs.append(prec); recs.append(rec); accs.append(acc)
        results[name] = {
            "label": label,
            "f1": f1s, "auc": aucs, "precision": precs, "recall": recs, "accuracy": accs,
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    labels = [results[k]["label"] for k in ("tree", "forest", "mlp")]
    box_f1 = [results[k]["f1"] for k in ("tree", "forest", "mlp")]
    box_auc = [results[k]["auc"] for k in ("tree", "forest", "mlp")]
    bp1 = axes[0].boxplot(box_f1, labels=labels, patch_artist=True, widths=0.5)
    for p in bp1["boxes"]:
        p.set_facecolor("0.85"); p.set_edgecolor("black")
    for p in bp1["medians"]:
        p.set_color("black"); p.set_linewidth(1.5)
    axes[0].set_title(f"F1 по {N_SPLITS} фолдам (GroupKFold)")
    axes[0].set_ylabel("F1 (класс «поломка»)")
    bp2 = axes[1].boxplot(box_auc, labels=labels, patch_artist=True, widths=0.5)
    for p in bp2["boxes"]:
        p.set_facecolor("0.85"); p.set_edgecolor("black")
    for p in bp2["medians"]:
        p.set_color("black"); p.set_linewidth(1.5)
    axes[1].set_title(f"AUC по {N_SPLITS} фолдам (GroupKFold)")
    axes[1].set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "boxplot_f1_auc.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
    metrics_names = ["F1", "Precision", "Recall", "Accuracy", "AUC"]
    metric_keys   = ["f1", "precision", "recall", "accuracy", "auc"]
    x = np.arange(len(metrics_names))
    width = 0.27
    styles = [
        dict(color="0.20", hatch=""),
        dict(color="0.55", hatch="//"),
        dict(color="0.80", hatch="xx"),
    ]
    for i, k in enumerate(("tree", "forest", "mlp")):
        means = [float(np.mean(results[k][m])) for m in metric_keys]
        stds  = [float(np.std(results[k][m])) for m in metric_keys]
        ax.bar(x + (i - 1) * width, means, width=width, yerr=stds,
               label=results[k]["label"], edgecolor="black",
               color=styles[i]["color"], hatch=styles[i]["hatch"], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Значение метрики")
    ax.set_title(f"Среднее ± стандартное отклонение по {N_SPLITS} фолдам")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "сводная_диаграмма.png", bbox_inches="tight")
    plt.close(fig)

    lines = []
    lines.append("# Кросс-валидация по полётам (GroupKFold)\n")
    lines.append(f"\nЧисло фолдов: **{N_SPLITS}**. Группировка: по `flight_id` "
                 f"(окна одного полёта целиком в обучении или в тестовом фолде).\n")
    lines.append(f"Всего окон: **{len(y)}**, уникальных полётов: **{len(set(groups.tolist()))}**, "
                 f"норма: {int((y == 0).sum())}, поломка: {int((y == 1).sum())}.\n")
    lines.append("\nГиперпараметры для каждой модели взяты из ранее обученных "
                 "`models/method_*.pkl` (подбор был сделан на отдельной отложенной валидации).\n")
    lines.append("\n## Сводная таблица (среднее ± std по фолдам)\n\n")
    lines.append("| Модель | F1 | Precision | Recall | Accuracy | AUC |\n")
    lines.append("|--------|----|-----------|--------|----------|-----|\n")
    for k in ("tree", "forest", "mlp"):
        r = results[k]
        cells = []
        for m in ("f1", "precision", "recall", "accuracy", "auc"):
            v = np.array(r[m], dtype=float)
            cells.append(f"{v.mean():.3f} ± {v.std():.3f}")
        lines.append(f"| {r['label']} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} |\n")

    lines.append("\n## По фолдам\n")
    for k in ("tree", "forest", "mlp"):
        r = results[k]
        lines.append(f"\n### {r['label']}\n\n")
        lines.append("| fold | F1 | Precision | Recall | Accuracy | AUC |\n")
        lines.append("|------|----|-----------|--------|----------|-----|\n")
        for i in range(N_SPLITS):
            lines.append(
                f"| {i+1} | {r['f1'][i]:.3f} | {r['precision'][i]:.3f} | "
                f"{r['recall'][i]:.3f} | {r['accuracy'][i]:.3f} | {r['auc'][i]:.3f} |\n"
            )

    lines.append("\n## Рисунки\n")
    lines.append("- `images/cv/boxplot_f1_auc.png` — распределения F1 и AUC по фолдам.\n")
    lines.append("- `images/cv/сводная_диаграмма.png` — среднее значение метрик ± std.\n")

    (DOCS_DIR / "cross_validation.md").write_text("".join(lines), encoding="utf-8")
    print("\nсохранено:", DOCS_DIR / "cross_validation.md")


if __name__ == "__main__":
    main()
