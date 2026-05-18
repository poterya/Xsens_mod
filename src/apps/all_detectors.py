#!/usr/bin/env python3
"""
Один процесс: общий поток Xsens → бинарные методы детекции одновременно
(Калман, Random Forest, Decision Tree; режим --compare-set03 добавляет MLP).

Мультикласс-модели здесь не используются (для них есть dual_kalman_rf.py).

По Ctrl+C (или концу файла в --offline):
  - в binary_detection_logs/binary_<live|offline>_<ts>/ сохраняются:
      compare.png, vibration_log.csv, README.txt,
      kalman_series.csv,
      rf_binary_series.csv, rf_binary_smoothed.csv,
      tree_binary_series.csv, tree_binary_smoothed.csv;
  - программа спрашивает качество винта и (если поломан) позицию,
    копирует vibration_log.csv в:
      datasets/set_04/Normal_mod/simulation_<ts>/vibration_log.csv
      datasets/set_04/Deformed_mod/<position>/simulation_<ts>/vibration_log.csv
    и печатает + сохраняет точность трёх методов в metrics_after_label.txt.

Использование:
  python3 all_detectors.py
  python3 all_detectors.py --calib datasets/set_03
  python3 all_detectors.py --offline path/to/vibration_log.csv
  python3 all_detectors.py --compare-set03
    (офлайн: случайные логи из calib, четыре метода → txt-таблица)
"""
from __future__ import annotations

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


import os

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import argparse
import random
import time
import warnings
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn

    class _MLPDetector(nn.Module):
        def __init__(self, in_features: int = 80, dropout: float = 0.3) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[misc, assignment]
    _MLPDetector = None  # type: ignore[misc, assignment]
    _TORCH_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from common.feature_extraction import extract_features

from common.kalman_fault_detector import (
    KalmanFaultDetector,
    EWMA_ALPHA,
    Q_FACTOR,
    calibrate_from_normal,
)
from common.realtime_detector import (
    STEP,
    WINDOW_SIZE,
    load_model,
)

from common.dual_kalman_rf import (
    SMOOTH_WIN,
    _ask_quality_and_position,
    _export_to_set04,
    _has_gui_display,
    _resolve_model,
    _smoothed,
    _truth_label_from_decision,
)

OUTPUT_ROOT_NAME = "runs/binary_detection_logs"

DEFAULT_RF_BIN_CANDIDATES = (
    "models/propeller_fault_rf_binary_set03.pkl",
    "models/propeller_fault_rf_binary.pkl",
)
DEFAULT_TREE_BIN_CANDIDATES = (
    "models/propeller_fault_tree_binary_set03.pkl",
    "models/propeller_fault_tree_binary.pkl",
)
DEFAULT_MLP_BIN_CANDIDATES = (
    "models/mlp/propeller_fault_mlp_keras_set05.pt",
)
BINARY_LABELS = ("normal", "fault")

# Минимум строк в vibration_log.csv, чтобы была хотя бы одна точка классификации (STEP после заполнения окна).
_MIN_ROWS_COMPARE = WINDOW_SIZE + STEP - 1


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _binary_proba_fault(clf, x: np.ndarray, labels: list[str]) -> tuple[str, float, float]:
    proba = clf.predict_proba(x)[0]
    idx_fault = labels.index("fault") if "fault" in labels else int(np.argmax(proba))
    p_fault = float(proba[idx_fault])
    best = int(np.argmax(proba))
    return labels[best], float(proba[best]), p_fault


# ---------------------------------------------------------------------------
# График
# ---------------------------------------------------------------------------


def _save_compare_plot(
    out_path: Path,
    kalman_t: np.ndarray, kalman_p: np.ndarray,
    rf_bin_t: np.ndarray, rf_bin_pf: np.ndarray,
    tree_bin_t: np.ndarray, tree_bin_pf: np.ndarray,
    rf_bin_label_t: np.ndarray, rf_bin_label_idx: np.ndarray,
    tree_bin_label_t: np.ndarray, tree_bin_label_idx: np.ndarray,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    if len(kalman_t):
        ax1.plot(kalman_t, kalman_p, color="tab:blue", linewidth=1.2, label="Калман p_fault")
    if len(rf_bin_t):
        ax1.plot(rf_bin_t, rf_bin_pf, color="tab:red", linewidth=1.0, alpha=0.9, label="RF бинарный p_fault")
    if len(tree_bin_t):
        ax1.plot(tree_bin_t, tree_bin_pf, color="tab:purple", linewidth=1.0, alpha=0.9, label="Tree бинарный p_fault")
    ax1.axhline(0.5, color="0.4", linestyle="--", linewidth=1, label="порог 0.5")
    ax1.set_ylabel("p(FAULT)")
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Калман + RF (бинарный) + Tree (бинарный) — одна сессия")

    if len(kalman_t):
        k_bin = (kalman_p >= 0.5).astype(float)
        ax2.step(kalman_t, k_bin, where="post", color="tab:blue", linewidth=1.2, label="Калман FAULT")
    if len(rf_bin_label_t):
        ax2.step(
            rf_bin_label_t, rf_bin_label_idx, where="post",
            color="tab:red", linewidth=1.0, label="RF бинарный (label)",
        )
    if len(tree_bin_label_t):
        ax2.step(
            tree_bin_label_t, tree_bin_label_idx, where="post",
            color="tab:purple", linewidth=1.0, label="Tree бинарный (label)",
        )
    ax2.set_ylabel("бинарно\n0=норма 1=поломка")
    ax2.set_xlabel("время, с")
    ax2.set_ylim(-0.1, 1.15)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if not out_path.is_file() or out_path.stat().st_size < 100:
        print(f"[!] Не удалось сохранить график: {out_path}", file=sys.stderr)
    else:
        print(f"График сохранён: {out_path.resolve()}")
    if _has_gui_display():
        try:
            plt.show(block=False)
            plt.pause(0.05)
        except Exception:
            pass
    plt.close(fig)


# ---------------------------------------------------------------------------
# Запись CSV/README
# ---------------------------------------------------------------------------


def _write_series(out_dir: Path, name: str, header: str, rows) -> None:
    with open(out_dir / name, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def _write_vibration_log(out_dir: Path, vib_rows) -> Path:
    p = out_dir / "vibration_log.csv"
    with open(p, "w") as f:
        f.write("time_seconds,total_vibration,rms_x,rms_y,rms_z\n")
        for t, total, rx, ry, rz in vib_rows:
            f.write(f"{t:.3f},{total:.6f},{rx:.6f},{ry:.6f},{rz:.6f}\n")
    return p


def _write_all_logs(
    out_dir: Path,
    kalman_log,
    rf_bin_log, rf_bin_smooth_log,
    tree_bin_log, tree_bin_smooth_log,
) -> None:
    _write_series(
        out_dir, "kalman_series.csv", "time_s,p_fault,fault_bin",
        ((f"{t:.4f}", f"{p:.6f}", b) for t, p, b in kalman_log),
    )
    _write_series(
        out_dir, "rf_binary_series.csv", "time_s,label,confidence,p_fault",
        ((f"{t:.4f}", lbl, f"{c:.6f}", f"{pf:.6f}") for t, lbl, c, pf in rf_bin_log),
    )
    _write_series(
        out_dir, "rf_binary_smoothed.csv", "time_s,label,confidence",
        ((f"{t:.4f}", lbl, f"{c:.6f}") for t, lbl, c in rf_bin_smooth_log),
    )
    _write_series(
        out_dir, "tree_binary_series.csv", "time_s,label,confidence,p_fault",
        ((f"{t:.4f}", lbl, f"{c:.6f}", f"{pf:.6f}") for t, lbl, c, pf in tree_bin_log),
    )
    _write_series(
        out_dir, "tree_binary_smoothed.csv", "time_s,label,confidence",
        ((f"{t:.4f}", lbl, f"{c:.6f}") for t, lbl, c in tree_bin_smooth_log),
    )
    with open(out_dir / "README.txt", "w") as f:
        f.write("Калман: бинарный p_fault (FAULT если >= 0.5)\n")
        f.write("Бинарный RF/Tree: " + ", ".join(BINARY_LABELS) + "\n")
        f.write("compare.png: всегда сохраняется на диск (Agg при отсутствии DISPLAY)\n")
        f.write(
            "metrics_after_label.txt: появляется после ответов n/d в консоли "
            "(точность трёх бинарных методов).\n"
        )


# ---------------------------------------------------------------------------
# Метрики после разметки
# ---------------------------------------------------------------------------


def _binary_acc(log, truth_is_fault: bool) -> tuple[int, int, float | None]:
    if not log:
        return 0, 0, None
    want = "fault" if truth_is_fault else "normal"
    ok = sum(1 for r in log if r[1] == want)
    return ok, len(log), ok / len(log)


def _kalman_acc(kalman_log, truth_is_fault: bool) -> tuple[int, int, float | None]:
    if not kalman_log:
        return 0, 0, None
    want = 1 if truth_is_fault else 0
    ok = sum(1 for r in kalman_log if r[2] == want)
    return ok, len(kalman_log), ok / len(kalman_log)


def _print_and_save_metrics(
    out_dir: Path, truth_label: str,
    kalman_log,
    rf_bin_log, rf_bin_smooth_log,
    tree_bin_log, tree_bin_smooth_log,
) -> None:
    truth_fault = truth_label != "normal"
    truth_bin = "fault" if truth_fault else "normal"
    lines: list[str] = [
        f"Истина (ваша разметка): {truth_label}  →  бинарно: {truth_bin}",
        "",
    ]

    def _fmt(name: str, ok_n_acc) -> str:
        ok_, n_, acc_ = ok_n_acc
        return (
            f"{name}: {ok_}/{n_}"
            + (f" = {100 * acc_:.1f}%" if acc_ is not None else " (нет точек)")
        )

    lines.append(_fmt("Калман (бинарно)", _kalman_acc(kalman_log, truth_fault)))
    lines.append(_fmt("RF бинарный, сырой", _binary_acc(rf_bin_log, truth_fault)))
    lines.append(_fmt("RF бинарный, сглаж.", _binary_acc(rf_bin_smooth_log, truth_fault)))
    lines.append(_fmt("Tree бинарный, сырой", _binary_acc(tree_bin_log, truth_fault)))
    lines.append(_fmt("Tree бинарный, сглаж.", _binary_acc(tree_bin_smooth_log, truth_fault)))

    print("\n=== Точность по вашей разметке ===")
    for ln in lines:
        print(ln)
    metrics_path = out_dir / "metrics_after_label.txt"
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nМетрики записаны: {metrics_path.resolve()}")


def _hint_from_logs(rf_bin_smooth_log, tree_bin_smooth_log, kalman_log) -> str:
    def _top(log):
        labels = [x[1] for x in log]
        return Counter(labels).most_common(1)[0][0] if labels else "—"

    rf_b = _top(rf_bin_smooth_log)
    tree_b = _top(tree_bin_smooth_log)
    if kalman_log:
        frac = sum(b for _, _, b in kalman_log) / len(kalman_log)
        kal = f"FAULT доля {frac:.2f}"
    else:
        kal = "нет"
    return f"RF бин «{rf_b}», Tree бин «{tree_b}»; Калман: {kal}"


# ---------------------------------------------------------------------------
# Сохранение и экспорт
# ---------------------------------------------------------------------------


def _do_plot(
    out_dir: Path,
    kalman_log,
    rf_bin_log, rf_bin_smooth_log,
    tree_bin_log, tree_bin_smooth_log,
) -> None:
    if not (kalman_log or rf_bin_log or tree_bin_log):
        return

    def _arr(log, *cols):
        if not log:
            return tuple(np.array([]) for _ in cols)
        return tuple(np.array([r[c] for r in log]) for c in cols)

    kt, kp = _arr(kalman_log, 0, 1)
    rf_bin_t, rf_bin_pf = _arr(rf_bin_log, 0, 3)
    tr_bin_t, tr_bin_pf = _arr(tree_bin_log, 0, 3)

    def _bin_label_array(log):
        if not log:
            return np.array([]), np.array([])
        return (
            np.array([r[0] for r in log]),
            np.array([0 if r[1] == "normal" else 1 for r in log]),
        )

    rfb_t, rfb_idx = _bin_label_array(rf_bin_smooth_log or rf_bin_log)
    trb_t, trb_idx = _bin_label_array(tree_bin_smooth_log or tree_bin_log)

    _save_compare_plot(
        out_dir / "compare.png",
        kt, kp,
        rf_bin_t, rf_bin_pf,
        tr_bin_t, tr_bin_pf,
        rfb_t, rfb_idx,
        trb_t, trb_idx,
    )


def _save_artifacts(
    base: Path, out_dir_name: str, vib_rows,
    kalman_log,
    rf_bin_log, rf_bin_smooth_log,
    tree_bin_log, tree_bin_smooth_log,
) -> Path:
    out_dir = base / OUTPUT_ROOT_NAME / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_all_logs(
        out_dir, kalman_log,
        rf_bin_log, rf_bin_smooth_log,
        tree_bin_log, tree_bin_smooth_log,
    )
    _write_vibration_log(out_dir, vib_rows)
    _do_plot(
        out_dir, kalman_log,
        rf_bin_log, rf_bin_smooth_log,
        tree_bin_log, tree_bin_smooth_log,
    )
    return out_dir


def _maybe_export(
    base: Path, out_dir: Path,
    kalman_log,
    rf_bin_log, rf_bin_smooth_log,
    tree_bin_log, tree_bin_smooth_log,
) -> None:
    src_csv = out_dir / "vibration_log.csv"
    if not src_csv.exists():
        print(f"[!] vibration_log.csv не найден в {out_dir}, экспорт пропущен.")
        return
    hint = _hint_from_logs(rf_bin_smooth_log, tree_bin_smooth_log, kalman_log)
    decision = _ask_quality_and_position(default_hint=hint)
    if decision is None:
        print("Экспорт в set_04 пропущен.")
        print("\nРазметка не задана — точность по истине не считается.")
        return
    quality, position = decision
    try:
        dst = _export_to_set04(base, src_csv, out_dir.name, quality, position)
    except KeyboardInterrupt:
        print("\nОтменено пользователем.")
        return
    where = "Normal_mod" if quality == "normal" else f"Deformed_mod/{position}"
    print(f"\nСохранено в datasets/set_04/{where}: {dst}")
    truth = _truth_label_from_decision(quality, position)
    _print_and_save_metrics(
        out_dir, truth,
        kalman_log,
        rf_bin_log, rf_bin_smooth_log,
        tree_bin_log, tree_bin_smooth_log,
    )


# ---------------------------------------------------------------------------
# Live и offline режимы
# ---------------------------------------------------------------------------


def _process_window(
    feats: dict,
    rf_bin_clf, rf_bin_features, rf_bin_labels,
    tree_bin_clf, tree_bin_features, tree_bin_labels,
):
    def vec(feature_names):
        return np.array([[feats.get(n, 0.0) for n in feature_names]], dtype=float)

    rfb_lbl, rfb_conf, rfb_pf = _binary_proba_fault(
        rf_bin_clf, vec(rf_bin_features), rf_bin_labels,
    )
    trb_lbl, trb_conf, trb_pf = _binary_proba_fault(
        tree_bin_clf, vec(tree_bin_features), tree_bin_labels,
    )
    return (rfb_lbl, rfb_conf, rfb_pf), (trb_lbl, trb_conf, trb_pf)


def _truth_fault_from_set03_csv_path(csv_path: Path) -> bool | None:
    parts = csv_path.resolve().parts
    if "Normal_mod" in parts:
        return False
    if "Deformed_mod" in parts:
        return True
    return None


def _vibration_csv_row_count(path: Path) -> int:
    with open(path, "rb") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


def _eligible_compare_logs(calib_root: Path) -> list[tuple[Path, bool]]:
    out: list[tuple[Path, bool]] = []
    root = calib_root.resolve()
    if not root.is_dir():
        return out
    for p in root.rglob("vibration_log.csv"):
        tf = _truth_fault_from_set03_csv_path(p)
        if tf is None:
            continue
        if _vibration_csv_row_count(p) < _MIN_ROWS_COMPARE:
            continue
        out.append((p.resolve(), tf))
    return out


def _load_mlp_torch_bundle(model_path: Path) -> dict:
    if not _TORCH_AVAILABLE or _MLPDetector is None:
        raise RuntimeError("Нужен пакет torch для MLP.")
    model = _MLPDetector(in_features=80)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    z = np.zeros(WINDOW_SIZE, dtype=float)
    feature_names = list(extract_features(z, z, z, z).keys())
    return {
        "classifier": model,
        "feature_names": feature_names,
        "class_labels": ["normal", "fault"],
    }


def _per_step_accuracy_one_log(
    csv_path: Path,
    truth_fault: bool,
    mu: np.ndarray,
    R: np.ndarray,
    rf_bin_b: dict,
    tree_bin_b: dict,
    mlp_bundle: dict,
) -> dict[str, float]:
    """Доля верных бинарных ответов на шагах классификации (как в --offline, без сглаживания)."""
    rf_bin_clf = rf_bin_b["classifier"]
    rf_bin_features = list(rf_bin_b["feature_names"])
    rf_bin_labels = list(rf_bin_b["class_labels"])
    tree_bin_clf = tree_bin_b["classifier"]
    tree_bin_features = list(tree_bin_b["feature_names"])
    tree_bin_labels = list(tree_bin_b["class_labels"])
    mlp_model = mlp_bundle["classifier"]
    mlp_feature_names = list(mlp_bundle["feature_names"])

    df = pd.read_csv(csv_path)
    cols = ["total_vibration", "rms_x", "rms_y", "rms_z"]
    if not set(cols).issubset(df.columns):
        return {"kalman": float("nan"), "tree": float("nan"), "rf": float("nan"), "mlp": float("nan")}

    truth_lbl = "fault" if truth_fault else "normal"
    truth_kal = 1 if truth_fault else 0

    kalm = KalmanFaultDetector(mu, R, q_factor=Q_FACTOR, ewma_alpha=EWMA_ALPHA)
    vib_total: deque = deque(maxlen=WINDOW_SIZE)
    vib_x: deque = deque(maxlen=WINDOW_SIZE)
    vib_y: deque = deque(maxlen=WINDOW_SIZE)
    vib_z: deque = deque(maxlen=WINDOW_SIZE)
    step_i = 0

    k_ok = k_tot = 0
    rf_ok = rf_tot = 0
    tr_ok = tr_tot = 0
    m_ok = m_tot = 0

    def vec(feats: dict, feature_names: list[str]) -> np.ndarray:
        return np.array([[feats.get(n, 0.0) for n in feature_names]], dtype=float)

    for row_idx, (_, row) in enumerate(df.iterrows()):
        z = row[cols].to_numpy(dtype=float)
        vib_total.append(z[0]); vib_x.append(z[1]); vib_y.append(z[2]); vib_z.append(z[3])
        out = kalm.update(z)

        if len(vib_total) < WINDOW_SIZE:
            continue
        step_i += 1
        if step_i % STEP != 0:
            continue

        feats = extract_features(
            np.fromiter(vib_total, dtype=float, count=WINDOW_SIZE),
            np.fromiter(vib_x, dtype=float, count=WINDOW_SIZE),
            np.fromiter(vib_y, dtype=float, count=WINDOW_SIZE),
            np.fromiter(vib_z, dtype=float, count=WINDOW_SIZE),
        )
        if int(out["p_fault"] >= 0.5) == truth_kal:
            k_ok += 1
        k_tot += 1

        rfb_lbl, _, _ = _binary_proba_fault(
            rf_bin_clf, vec(feats, rf_bin_features), rf_bin_labels,
        )
        if rfb_lbl == truth_lbl:
            rf_ok += 1
        rf_tot += 1

        trb_lbl, _, _ = _binary_proba_fault(
            tree_bin_clf, vec(feats, tree_bin_features), tree_bin_labels,
        )
        if trb_lbl == truth_lbl:
            tr_ok += 1
        tr_tot += 1

        x_t = torch.tensor(  # type: ignore[union-attr]
            [[float(feats.get(n, 0.0)) for n in mlp_feature_names]],
            dtype=torch.float32,
        )
        with torch.no_grad():  # type: ignore[union-attr]
            p_fault = float(torch.sigmoid(mlp_model(x_t))[0].item())
        m_lbl = "fault" if p_fault >= 0.5 else "normal"
        if m_lbl == truth_lbl:
            m_ok += 1
        m_tot += 1

    def acc(ok: int, tot: int) -> float:
        return ok / tot if tot else float("nan")

    return {
        "kalman": acc(k_ok, k_tot),
        "tree": acc(tr_ok, tr_tot),
        "rf": acc(rf_ok, rf_tot),
        "mlp": acc(m_ok, m_tot),
    }


def _write_table11_methods_txt(
    out_path: Path,
    rows: list[tuple[int, float, float, float, float]],
    mean_row: tuple[float, float, float, float],
) -> None:
    """Таблица: № лога, Калман, Decision Tree, Random Forest, MLP; строка Среднее; подпись таблицы 11."""
    sep = "\t"
    lines = [
        sep.join(("№ лога", "Калман", "Decision Tree", "Random Forest", "MLP")),
    ]
    for log_i, kal, tree, rf, mlp in rows:
        lines.append(sep.join((
            str(log_i),
            f"{kal:.3f}",
            f"{tree:.3f}",
            f"{rf:.3f}",
            f"{mlp:.3f}",
        )))
    mk, mtd, mrf, mm = mean_row
    lines.append(sep.join((
        "Среднее",
        f"{mk:.3f}",
        f"{mtd:.3f}",
        f"{mrf:.3f}",
        f"{mm:.3f}",
    )))
    lines.append("")
    lines.append("Таблица 11. Сравнение методов на одинаковых рандомных логах")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_offline_compare_set03(
    base: Path,
    calib_base: Path,
    rf_bin_path: Path,
    tree_bin_path: Path,
    mlp_path: Path,
    out_txt: Path,
    n_logs: int,
    seed: int,
) -> Path:
    if not _TORCH_AVAILABLE:
        print("Установите torch (pip install torch) для режима --compare-set03.", file=sys.stderr)
        sys.exit(1)

    normal_dir = calib_base / "Normal_mod"
    if not normal_dir.is_dir():
        print(f"Нет каталога калибровки: {normal_dir}", file=sys.stderr)
        sys.exit(1)

    eligible = _eligible_compare_logs(calib_base)
    if not eligible:
        print(
            f"Нет подходящих vibration_log.csv в {calib_base} "
            f"(нужны Normal_mod/Deformed_mod и ≥ {_MIN_ROWS_COMPARE} строк данных).",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = random.Random(seed)
    k = min(max(1, n_logs), len(eligible))
    chosen = rng.sample(eligible, k=k)

    print(f"RF бинарный:   {rf_bin_path}")
    print(f"Tree бинарный: {tree_bin_path}")
    print(f"MLP:           {mlp_path}")
    print(f"Калман μ,R по: {normal_dir}")
    print(f"Логов в выборке: {k}, seed={seed}")

    mu, R = calibrate_from_normal(normal_dir)
    rf_bin_b = load_model(rf_bin_path)
    tree_bin_b = load_model(tree_bin_path)
    mlp_bundle = _load_mlp_torch_bundle(mlp_path)

    table_rows: list[tuple[int, float, float, float, float]] = []
    sum_k = sum_t = sum_rf = sum_m = 0.0

    for j, (csv_p, truth_fault) in enumerate(chosen, start=1):
        accs = _per_step_accuracy_one_log(
            csv_p, truth_fault, mu, R, rf_bin_b, tree_bin_b, mlp_bundle,
        )
        kal, tree, rf, mlp = accs["kalman"], accs["tree"], accs["rf"], accs["mlp"]
        table_rows.append((j, kal, tree, rf, mlp))
        sum_k += kal
        sum_t += tree
        sum_rf += rf
        sum_m += mlp
        try:
            rel = str(csv_p.relative_to(base))
        except ValueError:
            rel = str(csv_p)
        print(f"  [{j}] {rel}  kal={kal:.3f} tree={tree:.3f} rf={rf:.3f} mlp={mlp:.3f}")

    inv_k = 1.0 / k
    mean_row = (sum_k * inv_k, sum_t * inv_k, sum_rf * inv_k, sum_m * inv_k)
    out_txt = out_txt.resolve()
    _write_table11_methods_txt(out_txt, table_rows, mean_row)
    print(f"\nТаблица записана: {out_txt}")
    return out_txt


class _RMSWindow:
    """Минимальный аналог RealtimeFaultDetector без классификатора:
    держит окно ускорений, выдаёт скользящие RMS и total."""

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        self.window_size = window_size
        self.acc_x: deque = deque(maxlen=window_size)
        self.acc_y: deque = deque(maxlen=window_size)
        self.acc_z: deque = deque(maxlen=window_size)
        self.vib_total: deque = deque(maxlen=window_size)
        self.vib_x: deque = deque(maxlen=window_size)
        self.vib_y: deque = deque(maxlen=window_size)
        self.vib_z: deque = deque(maxlen=window_size)

    def add(self, ax: float, ay: float, az: float) -> None:
        self.acc_x.append(ax); self.acc_y.append(ay); self.acc_z.append(az)
        if len(self.acc_x) >= self.window_size:
            ax_arr = np.fromiter(self.acc_x, dtype=float, count=self.window_size)
            ay_arr = np.fromiter(self.acc_y, dtype=float, count=self.window_size)
            az_arr = np.fromiter(self.acc_z, dtype=float, count=self.window_size)
            mx, my, mz = ax_arr.mean(), ay_arr.mean(), az_arr.mean()
            rms_x = float(np.sqrt(np.mean((ax_arr - mx) ** 2)))
            rms_y = float(np.sqrt(np.mean((ay_arr - my) ** 2)))
            rms_z = float(np.sqrt(np.mean((az_arr - mz) ** 2)))
            total = float(np.sqrt(rms_x ** 2 + rms_y ** 2 + rms_z ** 2))
            self.vib_total.append(total)
            self.vib_x.append(rms_x)
            self.vib_y.append(rms_y)
            self.vib_z.append(rms_z)


def run_live(
    base: Path, calib_base: Path,
    rf_bin_path: Path, tree_bin_path: Path,
) -> Path:
    from xsens.DataPacketParser import DataPacketParser, XsDataPacket
    from xsens.SerialHandler import SerialHandler
    from xsens.XbusPacket import XbusPacket

    normal_dir = calib_base / "Normal_mod"
    if not normal_dir.is_dir():
        print(f"Нет {normal_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"RF бинарный:  {rf_bin_path}")
    print(f"Tree бинарный:{tree_bin_path}")
    print(f"Калибровка Калмана по: {normal_dir}")

    rf_bin_b = load_model(rf_bin_path)
    tree_bin_b = load_model(tree_bin_path)
    rf_bin_clf = rf_bin_b["classifier"]
    rf_bin_features = list(rf_bin_b["feature_names"])
    rf_bin_labels = list(rf_bin_b["class_labels"])
    tree_bin_clf = tree_bin_b["classifier"]
    tree_bin_features = list(tree_bin_b["feature_names"])
    tree_bin_labels = list(tree_bin_b["class_labels"])

    mu, R = calibrate_from_normal(normal_dir)
    kalm = KalmanFaultDetector(mu, R, q_factor=Q_FACTOR, ewma_alpha=EWMA_ALPHA)

    rmsw = _RMSWindow(WINDOW_SIZE)

    vib_rows: list[tuple[float, float, float, float, float]] = []
    kalman_log: list[tuple[float, float, int]] = []
    rf_bin_log: list[tuple[float, str, float, float]] = []
    tree_bin_log: list[tuple[float, str, float, float]] = []

    rfb_buf: deque = deque(maxlen=SMOOTH_WIN)
    trb_buf: deque = deque(maxlen=SMOOTH_WIN)
    rf_bin_smooth_log: list[tuple[float, str, float]] = []
    tree_bin_smooth_log: list[tuple[float, str, float]] = []

    serial = SerialHandler("/dev/ttyUSB0", 115200)
    serial.send_with_checksum(bytes.fromhex("FA FF 30 00"))
    time.sleep(0.1)
    serial.send_with_checksum(bytes.fromhex(
        "FA FF C0 20 10 20 FF FF 10 60 FF FF "
        "20 30 00 64 40 20 00 64 40 30 00 64 "
        "80 20 00 64 C0 20 00 64 E0 20 FF FF"
    ))
    time.sleep(0.1)
    serial.send_with_checksum(bytes.fromhex("FA FF 10 00"))
    print("Measurement mode. Ctrl+C для остановки.")

    step_counter = [0]
    start_time = time.time()

    def on_packet(raw_packet):
        xbus = XsDataPacket()
        DataPacketParser.parse_data_packet(raw_packet, xbus)
        if not xbus.accAvailable:
            return
        n_before = len(rmsw.vib_total)
        rmsw.add(xbus.acc[0], xbus.acc[1], xbus.acc[2])
        if len(rmsw.vib_total) > n_before:
            t_rel = time.time() - start_time
            total = float(rmsw.vib_total[-1])
            rx = float(rmsw.vib_x[-1])
            ry = float(rmsw.vib_y[-1])
            rz = float(rmsw.vib_z[-1])
            vib_rows.append((t_rel, total, rx, ry, rz))
            out = kalm.update(np.array([total, rx, ry, rz], dtype=float))
            kalman_log.append((t_rel, float(out["p_fault"]), int(out["p_fault"] >= 0.5)))

        step_counter[0] += 1
        if step_counter[0] % STEP != 0:
            return
        if len(rmsw.vib_total) < rmsw.window_size:
            return

        feats = extract_features(
            np.fromiter(rmsw.vib_total, dtype=float, count=rmsw.window_size),
            np.fromiter(rmsw.vib_x, dtype=float, count=rmsw.window_size),
            np.fromiter(rmsw.vib_y, dtype=float, count=rmsw.window_size),
            np.fromiter(rmsw.vib_z, dtype=float, count=rmsw.window_size),
        )
        rfb, trb = _process_window(
            feats,
            rf_bin_clf, rf_bin_features, rf_bin_labels,
            tree_bin_clf, tree_bin_features, tree_bin_labels,
        )
        t_s = time.time() - start_time
        rf_bin_log.append((t_s, rfb[0], rfb[1], rfb[2]))
        tree_bin_log.append((t_s, trb[0], trb[1], trb[2]))
        rfb_buf.append((rfb[0], rfb[1])); trb_buf.append((trb[0], trb[1]))
        for buf, dest in (
            (rfb_buf, rf_bin_smooth_log),
            (trb_buf, tree_bin_smooth_log),
        ):
            sm = _smoothed(buf)
            if sm is not None:
                dest.append((t_s, sm[0], sm[1]))

    packet = XbusPacket(on_data_available=on_packet)
    try:
        while True:
            b = serial.read_byte()
            if b:
                packet.feed_byte(b)
    except KeyboardInterrupt:
        print("\nОстановлено.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _save_artifacts(
        base, f"binary_live_{ts}", vib_rows,
        kalman_log,
        rf_bin_log, rf_bin_smooth_log,
        tree_bin_log, tree_bin_smooth_log,
    )
    print(f"Артефакты: {out_dir}")
    _maybe_export(
        base, out_dir,
        kalman_log,
        rf_bin_log, rf_bin_smooth_log,
        tree_bin_log, tree_bin_smooth_log,
    )
    return out_dir


def run_offline(
    base: Path, calib_base: Path, csv_path: Path,
    rf_bin_path: Path, tree_bin_path: Path,
) -> Path:
    normal_dir = calib_base / "Normal_mod"
    print(f"RF бинарный:  {rf_bin_path}")
    print(f"Tree бинарный:{tree_bin_path}")
    print(f"Калибровка Калмана по: {normal_dir}")

    rf_bin_b = load_model(rf_bin_path)
    tree_bin_b = load_model(tree_bin_path)
    rf_bin_clf = rf_bin_b["classifier"]
    rf_bin_features = list(rf_bin_b["feature_names"])
    rf_bin_labels = list(rf_bin_b["class_labels"])
    tree_bin_clf = tree_bin_b["classifier"]
    tree_bin_features = list(tree_bin_b["feature_names"])
    tree_bin_labels = list(tree_bin_b["class_labels"])

    mu, R = calibrate_from_normal(normal_dir)
    kalm = KalmanFaultDetector(mu, R, q_factor=Q_FACTOR, ewma_alpha=EWMA_ALPHA)

    df = pd.read_csv(csv_path)
    cols = ["total_vibration", "rms_x", "rms_y", "rms_z"]
    if not set(cols).issubset(df.columns):
        print(f"В CSV нужны колонки: {cols}", file=sys.stderr)
        sys.exit(1)
    times = (
        df["time_seconds"].to_numpy(dtype=float)
        if "time_seconds" in df.columns
        else np.arange(len(df), dtype=float) * 0.01
    )

    vib_rows: list[tuple[float, float, float, float, float]] = []
    kalman_log: list[tuple[float, float, int]] = []
    rf_bin_log: list[tuple[float, str, float, float]] = []
    tree_bin_log: list[tuple[float, str, float, float]] = []

    rfb_buf: deque = deque(maxlen=SMOOTH_WIN)
    trb_buf: deque = deque(maxlen=SMOOTH_WIN)
    rf_bin_smooth_log: list[tuple[float, str, float]] = []
    tree_bin_smooth_log: list[tuple[float, str, float]] = []

    vib_total: deque = deque(maxlen=WINDOW_SIZE)
    vib_x: deque = deque(maxlen=WINDOW_SIZE)
    vib_y: deque = deque(maxlen=WINDOW_SIZE)
    vib_z: deque = deque(maxlen=WINDOW_SIZE)
    step_i = 0

    for row_idx, (_, row) in enumerate(df.iterrows()):
        z = row[cols].to_numpy(dtype=float)
        t_wall = float(times[row_idx]) if row_idx < len(times) else row_idx * 0.01
        vib_total.append(z[0]); vib_x.append(z[1]); vib_y.append(z[2]); vib_z.append(z[3])
        vib_rows.append((t_wall, float(z[0]), float(z[1]), float(z[2]), float(z[3])))

        out = kalm.update(z)
        kalman_log.append((t_wall, float(out["p_fault"]), int(out["p_fault"] >= 0.5)))

        if len(vib_total) < WINDOW_SIZE:
            continue
        step_i += 1
        if step_i % STEP != 0:
            continue
        feats = extract_features(
            np.fromiter(vib_total, dtype=float, count=WINDOW_SIZE),
            np.fromiter(vib_x, dtype=float, count=WINDOW_SIZE),
            np.fromiter(vib_y, dtype=float, count=WINDOW_SIZE),
            np.fromiter(vib_z, dtype=float, count=WINDOW_SIZE),
        )
        rfb, trb = _process_window(
            feats,
            rf_bin_clf, rf_bin_features, rf_bin_labels,
            tree_bin_clf, tree_bin_features, tree_bin_labels,
        )
        rf_bin_log.append((t_wall, rfb[0], rfb[1], rfb[2]))
        tree_bin_log.append((t_wall, trb[0], trb[1], trb[2]))
        rfb_buf.append((rfb[0], rfb[1])); trb_buf.append((trb[0], trb[1]))
        for buf, dest in (
            (rfb_buf, rf_bin_smooth_log),
            (trb_buf, tree_bin_smooth_log),
        ):
            sm = _smoothed(buf)
            if sm is not None:
                dest.append((t_wall, sm[0], sm[1]))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _save_artifacts(
        base, f"binary_offline_{ts}", vib_rows,
        kalman_log,
        rf_bin_log, rf_bin_smooth_log,
        tree_bin_log, tree_bin_smooth_log,
    )
    print(f"Артефакты: {out_dir}")
    _maybe_export(
        base, out_dir,
        kalman_log,
        rf_bin_log, rf_bin_smooth_log,
        tree_bin_log, tree_bin_smooth_log,
    )
    return out_dir


def main() -> None:
    here = _PROJECT_ROOT
    parser = argparse.ArgumentParser(
        description="Калман + RF + Tree (бинарные); экспорт в datasets/set_04",
    )
    parser.add_argument(
        "--base", type=Path, default=here,
        help=f"Корень проекта (модели, {OUTPUT_ROOT_NAME}, datasets/set_04)",
    )
    parser.add_argument(
        "--calib", type=Path, default=here / "datasets" / "set_03",
        help="Датасет с Normal_mod для калибровки Калмана",
    )
    parser.add_argument(
        "--offline", type=Path, default=None,
        help="Режим без Xsens: vibration_log.csv (как в main.py)",
    )
    parser.add_argument("--rf-binary-model", type=Path, default=None,
                        help=f"pkl с RF бинарный (по умолчанию: {', '.join(DEFAULT_RF_BIN_CANDIDATES)})")
    parser.add_argument("--tree-binary-model", type=Path, default=None,
                        help=f"pkl с Tree бинарный (по умолчанию: {', '.join(DEFAULT_TREE_BIN_CANDIDATES)})")
    parser.add_argument(
        "--compare-set03", action="store_true",
        help="Офлайн: случайная выборка логов из --calib (Normal_mod / Deformed_mod), "
        "сравнение Калман / Tree / RF / MLP → txt-таблица (нужен torch)",
    )
    parser.add_argument(
        "--compare-out", type=Path, default=None,
        help="Куда записать таблицу (по умолчанию: runs/compare_methods_offline/table11_methods_<ts>.txt)",
    )
    parser.add_argument("--compare-n", type=int, default=10, help="Число логов в выборке (по умолчанию 10)")
    parser.add_argument("--compare-seed", type=int, default=42, help="Seed для random.sample")
    parser.add_argument(
        "--mlp-binary-model", type=Path, default=None,
        help=f"Веса MLP .pt (по умолчанию: {', '.join(DEFAULT_MLP_BIN_CANDIDATES)})",
    )
    args = parser.parse_args()

    base = args.base.resolve()
    rf_bin_path = _resolve_model(base, args.rf_binary_model, DEFAULT_RF_BIN_CANDIDATES)
    tree_bin_path = _resolve_model(base, args.tree_binary_model, DEFAULT_TREE_BIN_CANDIDATES)

    if args.compare_set03 and args.offline is not None:
        parser.error("Нельзя одновременно --compare-set03 и --offline")

    if args.compare_set03:
        mlp_path = _resolve_model(base, args.mlp_binary_model, DEFAULT_MLP_BIN_CANDIDATES)
        out_txt = args.compare_out
        if out_txt is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_txt = base / "runs" / "compare_methods_offline" / f"table11_methods_{ts}.txt"
        else:
            out_txt = out_txt if out_txt.is_absolute() else base / out_txt
        run_offline_compare_set03(
            base,
            args.calib.resolve(),
            rf_bin_path,
            tree_bin_path,
            mlp_path,
            out_txt,
            args.compare_n,
            args.compare_seed,
        )
    elif args.offline is not None:
        run_offline(
            base, args.calib.resolve(), args.offline.resolve(),
            rf_bin_path, tree_bin_path,
        )
    else:
        run_live(base, args.calib.resolve(), rf_bin_path, tree_bin_path)


if __name__ == "__main__":
    main()
