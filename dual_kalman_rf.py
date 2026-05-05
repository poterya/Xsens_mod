#!/usr/bin/env python3
"""
Один процесс: общий поток Xsens → три метода детекции одновременно:
  1. Калман-фильтр (бинарный FAULT по Mahalanobis);
  2. Random Forest (мультикласс);
  3. Decision Tree (мультикласс).

По Ctrl+C (или концу файла в --offline):
  - в detection_logs/triple_<live|offline>_<ts>/ сохраняются:
      compare.png (всегда на диск), kalman_series.csv, rf_series.csv, rf_smoothed.csv,
      tree_series.csv, tree_smoothed.csv, vibration_log.csv, README.txt;
    после разметки в консоли: metrics_after_label.txt (точность трёх методов);
  - программа спрашивает качество винта и (если поломан) позицию
    и копирует vibration_log.csv в:
      datasets/set_04/Normal_mod/simulation_<ts>/vibration_log.csv
      datasets/set_04/Deformed_mod/<position>/simulation_<ts>/vibration_log.csv

Использование:
  python3 dual_kalman_rf.py
  python3 dual_kalman_rf.py --calib datasets/set_03
  python3 dual_kalman_rf.py --offline path/to/vibration_log.csv
  python3 dual_kalman_rf.py --rf-model propeller_fault_rf_set03.pkl \
                             --tree-model propeller_fault_tree_set03.pkl
"""
from __future__ import annotations

import os

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import argparse
import shutil
import sys
import time
import warnings
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from feature_extraction import extract_features

from kalman_fault_detector import (
    KalmanFaultDetector,
    EWMA_ALPHA,
    Q_FACTOR,
    calibrate_from_normal,
)
from realtime_detector import (
    RealtimeFaultDetector,
    STEP,
    WINDOW_SIZE,
    load_model,
)

POSITIONS: tuple[str, ...] = ("front_left", "front_right", "rear_left", "rear_right")
SMOOTH_WIN = 5

DEFAULT_RF_CANDIDATES = (
    "propeller_fault_rf_set03.pkl",
    "propeller_fault_rf.pkl",
    "propeller_fault_model.pkl",
)
DEFAULT_TREE_CANDIDATES = (
    "propeller_fault_tree_set03.pkl",
    "propeller_fault_tree.pkl",
)


def _has_gui_display() -> bool:
    return bool(
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
        or sys.platform == "darwin"
    )


def _label_to_idx(labels: list[str], name: str) -> int:
    try:
        return labels.index(name)
    except ValueError:
        return -1


def _resolve_model(base: Path, explicit: Path | None, candidates: tuple[str, ...]) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else (base / explicit)
        if not path.exists():
            raise SystemExit(f"Модель не найдена: {path}")
        return path
    for name in candidates:
        path = base / name
        if path.exists():
            return path
    raise SystemExit(
        "Не найдена ни одна модель из: " + ", ".join(candidates)
    )


def _smoothed(buf: deque) -> tuple[str, float] | None:
    if not buf:
        return None
    counts: Counter = Counter()
    best_conf: dict[str, float] = {}
    for lbl, c in buf:
        counts[lbl] += 1
        if c > best_conf.get(lbl, 0.0):
            best_conf[lbl] = c
    lbl, _ = max(counts.items(), key=lambda kv: (kv[1], best_conf[kv[0]]))
    return lbl, best_conf[lbl]


# ---------------------------------------------------------------------------
# Графики/логи
# ---------------------------------------------------------------------------


def _plot_class_axis(
    ax,
    class_labels: list[str],
    raw_t: np.ndarray,
    raw_idx: np.ndarray,
    smooth_t: np.ndarray | None,
    smooth_idx: np.ndarray | None,
    title: str,
    color_raw: str,
    color_smooth: str,
) -> None:
    ax.step(raw_t, raw_idx, where="post", color=color_raw, linewidth=1.0, label="сырое")
    if smooth_t is not None and len(smooth_t) > 0:
        ax.step(
            smooth_t,
            smooth_idx,
            where="post",
            color=color_smooth,
            linewidth=1.2,
            alpha=0.85,
            label="сглаженное",
        )
    ax.set_yticks(range(len(class_labels)))
    ax.set_yticklabels(class_labels, fontsize=8)
    ax.set_ylabel(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_compare(
    class_labels: list[str],
    kalman_t: np.ndarray,
    kalman_p: np.ndarray,
    rf_t: np.ndarray,
    rf_idx: np.ndarray,
    rf_smooth_t: np.ndarray | None,
    rf_smooth_idx: np.ndarray | None,
    tree_t: np.ndarray,
    tree_idx: np.ndarray,
    tree_smooth_t: np.ndarray | None,
    tree_smooth_idx: np.ndarray | None,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    ax1.plot(kalman_t, kalman_p, color="tab:blue", linewidth=1.2, label="Калман p_fault")
    ax1.axhline(0.5, color="tab:red", linestyle="--", linewidth=1, label="порог 0.5")
    ax1.set_ylabel("Калман\np(FAULT)")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Калман + Random Forest + Decision Tree — одна сессия, один поток данных")

    _plot_class_axis(
        ax2, class_labels, rf_t, rf_idx, rf_smooth_t, rf_smooth_idx,
        title="RF класс", color_raw="tab:green", color_smooth="tab:orange",
    )
    _plot_class_axis(
        ax3, class_labels, tree_t, tree_idx, tree_smooth_t, tree_smooth_idx,
        title="Tree класс", color_raw="tab:purple", color_smooth="tab:olive",
    )

    k_bin = (kalman_p >= 0.5).astype(float) if len(kalman_p) else kalman_p
    ax4.step(kalman_t, k_bin, where="post", color="tab:blue", linewidth=1.2, label="Калман FAULT")
    if rf_smooth_t is not None and len(rf_smooth_t) > 0:
        ax4.step(
            rf_smooth_t,
            (rf_smooth_idx > 0).astype(float),
            where="post",
            color="tab:orange",
            linewidth=1.2,
            label="RF поломка (сглаж.)",
        )
    if tree_smooth_t is not None and len(tree_smooth_t) > 0:
        ax4.step(
            tree_smooth_t,
            (tree_smooth_idx > 0).astype(float),
            where="post",
            color="tab:olive",
            linewidth=1.2,
            label="Tree поломка (сглаж.)",
        )
    ax4.set_ylabel("бинарно\n0=норма 1=поломка")
    ax4.set_xlabel("время, с")
    ax4.set_ylim(-0.1, 1.15)
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)

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


def _write_series(out_dir: Path, name: str, header: str, rows) -> None:
    with open(out_dir / name, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def _write_logs(
    out_dir: Path,
    kalman_log,
    rf_log,
    rf_smooth_log,
    tree_log,
    tree_smooth_log,
    class_labels: list[str],
) -> None:
    _write_series(
        out_dir, "kalman_series.csv", "time_s,p_fault,fault_bin",
        ((f"{t:.4f}", f"{p:.6f}", b) for t, p, b in kalman_log),
    )
    _write_series(
        out_dir, "rf_series.csv", "time_s,label,confidence",
        ((f"{t:.4f}", lbl, f"{c:.6f}") for t, lbl, c in rf_log),
    )
    _write_series(
        out_dir, "rf_smoothed.csv", "time_s,label,confidence",
        ((f"{t:.4f}", lbl, f"{c:.6f}") for t, lbl, c in rf_smooth_log),
    )
    _write_series(
        out_dir, "tree_series.csv", "time_s,label,confidence",
        ((f"{t:.4f}", lbl, f"{c:.6f}") for t, lbl, c in tree_log),
    )
    _write_series(
        out_dir, "tree_smoothed.csv", "time_s,label,confidence",
        ((f"{t:.4f}", lbl, f"{c:.6f}") for t, lbl, c in tree_smooth_log),
    )
    with open(out_dir / "README.txt", "w") as f:
        f.write("Классы (RF и Tree): " + ", ".join(class_labels) + "\n")
        f.write("Калман: бинарный p_fault (FAULT если >= 0.5)\n")
        f.write("compare.png: всегда сохраняется на диск (Agg при отсутствии DISPLAY)\n")
        f.write(
            "metrics_after_label.txt: появляется после ответов n/d в консоли "
            "(точность Калмана + RF + Tree)\n"
        )


def _write_vibration_log(out_dir: Path, vib_rows) -> Path:
    p = out_dir / "vibration_log.csv"
    with open(p, "w") as f:
        f.write("time_seconds,total_vibration,rms_x,rms_y,rms_z\n")
        for t, total, rx, ry, rz in vib_rows:
            f.write(f"{t:.3f},{total:.6f},{rx:.6f},{ry:.6f},{rz:.6f}\n")
    return p


def _do_plot(
    out_dir: Path,
    class_labels: list[str],
    kalman_log,
    rf_log,
    rf_smooth_log,
    tree_log,
    tree_smooth_log,
) -> None:
    if not kalman_log and not rf_log and not tree_log:
        return

    def _t_p(log, idx_field):
        if not log:
            return np.array([]), np.array([])
        return (
            np.array([x[0] for x in log]),
            np.array([x[idx_field] for x in log]),
        )

    kt = np.array([x[0] for x in kalman_log]) if kalman_log else np.array([])
    kp = np.array([x[1] for x in kalman_log]) if kalman_log else np.array([])

    def _label_array(log):
        if not log:
            return np.array([]), np.array([])
        t = np.array([x[0] for x in log])
        idx = np.array([_label_to_idx(class_labels, x[1]) for x in log])
        return t, idx

    rf_t, rf_idx = _label_array(rf_log)
    rf_st, rf_sidx = _label_array(rf_smooth_log)
    tr_t, tr_idx = _label_array(tree_log)
    tr_st, tr_sidx = _label_array(tree_smooth_log)

    plot_compare(
        class_labels,
        kt, kp,
        rf_t, rf_idx, rf_st if len(rf_st) else None, rf_sidx if len(rf_sidx) else None,
        tr_t, tr_idx, tr_st if len(tr_st) else None, tr_sidx if len(tr_sidx) else None,
        out_dir / "compare.png",
    )


# ---------------------------------------------------------------------------
# Запрос качества и экспорт в set_04
# ---------------------------------------------------------------------------


def _ask_yn(prompt: str) -> str:
    while True:
        ans = input(prompt).strip().lower()
        if ans:
            return ans
        print("Пустой ввод, попробуйте ещё раз.")


def _ask_quality_and_position(
    default_hint: str | None = None, dataset_name: str = "set_04"
) -> tuple[str, str | None] | None:
    print(f"\n=== Разметка полёта (для datasets/{dataset_name}) ===")
    if default_hint:
        print(f"Подсказка от моделей: {default_hint}")
    while True:
        ans = _ask_yn(
            "Качество винтов? [n] норма / [d] деформирован / [s] пропустить: "
        )
        if ans in ("n", "normal", "н"):
            return "normal", None
        if ans in ("d", "deformed", "д"):
            break
        if ans in ("s", "skip", "п"):
            return None
        print("Не понял. Введите n, d или s.")

    print("Позиция сломанного винта:")
    for i, p in enumerate(POSITIONS, 1):
        print(f"  {i}) {p}")
    while True:
        ans = _ask_yn("Введите 1-4 или имя позиции: ")
        if ans in {"1", "2", "3", "4"}:
            return "deformed", POSITIONS[int(ans) - 1]
        if ans in POSITIONS:
            return "deformed", ans
        print("Не понял.")


def _export_to_dataset(
    base: Path,
    src_csv: Path,
    out_dir_name: str,
    quality: str,
    position: str | None,
    dataset_subdir: str,
) -> Path:
    dataset_root = base / "datasets" / dataset_subdir
    if quality == "normal":
        target_root = dataset_root / "Normal_mod"
    else:
        if not position:
            raise SystemExit("Не указана позиция деформированного винта")
        target_root = dataset_root / "Deformed_mod" / position

    parts = out_dir_name.split("_")
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        ts = f"{parts[-2]}_{parts[-1]}"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    sim_dir = target_root / f"simulation_{ts}"
    while sim_dir.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_dir = target_root / f"simulation_{ts}"
        time.sleep(1)
    sim_dir.mkdir(parents=True, exist_ok=True)

    dst = sim_dir / "vibration_log.csv"
    shutil.copyfile(src_csv, dst)
    return dst


def _export_to_set04(
    base: Path, src_csv: Path, out_dir_name: str, quality: str, position: str | None,
) -> Path:
    return _export_to_dataset(base, src_csv, out_dir_name, quality, position, "set_04")


def _hint_from_logs(rf_smooth_log, tree_smooth_log, kalman_log) -> str:
    def _top(log):
        labels = [x[1] for x in log]
        return Counter(labels).most_common(1)[0][0] if labels else "—"

    rf_top = _top(rf_smooth_log)
    tree_top = _top(tree_smooth_log)
    if kalman_log:
        frac = sum(b for _, _, b in kalman_log) / len(kalman_log)
        kal = f"FAULT доля {frac:.2f}"
    else:
        kal = "нет"
    return f"RF доминирует «{rf_top}»; Tree доминирует «{tree_top}»; Калман: {kal}"


def _truth_label_from_decision(quality: str, position: str | None) -> str:
    if quality == "normal":
        return "normal"
    if not position:
        raise ValueError("position required for deformed")
    return position


def _multiclass_accuracy(
    log: list[tuple[float, str, float]], truth: str
) -> tuple[int, int, float | None]:
    if not log:
        return 0, 0, None
    ok = sum(1 for r in log if r[1] == truth)
    return ok, len(log), ok / len(log)


def _kalman_binary_accuracy(
    kalman_log: list[tuple[float, float, int]], truth_is_fault: bool,
) -> tuple[int, int, float | None]:
    if not kalman_log:
        return 0, 0, None
    want = 1 if truth_is_fault else 0
    ok = sum(1 for r in kalman_log if r[2] == want)
    return ok, len(kalman_log), ok / len(kalman_log)


def _print_and_save_session_metrics(
    out_dir: Path,
    truth_label: str,
    kalman_log: list[tuple[float, float, int]],
    rf_log: list[tuple[float, str, float]],
    rf_smooth_log: list[tuple[float, str, float]],
    tree_log: list[tuple[float, str, float]],
    tree_smooth_log: list[tuple[float, str, float]],
) -> None:
    truth_fault = truth_label != "normal"
    lines: list[str] = []
    lines.append(f"Истина (ваша разметка): {truth_label}")
    lines.append("")

    k_ok, k_n, k_acc = _kalman_binary_accuracy(kalman_log, truth_fault)
    lines.append(
        "Калман (бинарно: при норме ожидаем NORMAL, при поломке — FAULT): "
        f"{k_ok}/{k_n}"
        + (f" = {100 * k_acc:.1f}%" if k_acc is not None else " (нет точек)")
    )

    def _line(name: str, log: list[tuple[float, str, float]]) -> str:
        ok_, n_, acc_ = _multiclass_accuracy(log, truth_label)
        return (
            f"{name} (мультикласс, окно vs истина): {ok_}/{n_}"
            + (f" = {100 * acc_:.1f}%" if acc_ is not None else " (нет точек)")
        )

    lines.append(_line("RF сырой", rf_log))
    lines.append(_line("RF сглаж.", rf_smooth_log))
    lines.append(_line("Tree сырой", tree_log))
    lines.append(_line("Tree сглаж.", tree_smooth_log))

    print("\n=== Точность по вашей разметке ===")
    for ln in lines:
        print(ln)
    metrics_path = out_dir / "metrics_after_label.txt"
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nМетрики записаны: {metrics_path.resolve()}")


def _save_artifacts(
    base: Path,
    out_dir_name: str,
    vib_rows,
    kalman_log,
    rf_log,
    rf_smooth_log,
    tree_log,
    tree_smooth_log,
    class_labels: list[str],
) -> Path:
    out_dir = base / "detection_logs" / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_logs(
        out_dir,
        kalman_log,
        rf_log,
        rf_smooth_log,
        tree_log,
        tree_smooth_log,
        class_labels,
    )
    _write_vibration_log(out_dir, vib_rows)
    _do_plot(
        out_dir, class_labels, kalman_log, rf_log, rf_smooth_log, tree_log, tree_smooth_log,
    )
    return out_dir


def _maybe_export(
    base: Path,
    out_dir: Path,
    rf_smooth_log,
    tree_smooth_log,
    kalman_log,
    rf_log: list[tuple[float, str, float]],
    tree_log: list[tuple[float, str, float]],
) -> tuple[str, str | None] | None:
    src_csv = out_dir / "vibration_log.csv"
    if not src_csv.exists():
        print(f"[!] vibration_log.csv не найден в {out_dir}, экспорт пропущен.")
        return None
    hint = _hint_from_logs(rf_smooth_log, tree_smooth_log, kalman_log)
    decision = _ask_quality_and_position(default_hint=hint)
    if decision is None:
        print("Экспорт в set_04 пропущен.")
        print(
            "\nРазметка не задана — точность по истине не считается "
            "(выберите n или d при следующем запуске)."
        )
        return None
    quality, position = decision
    try:
        dst = _export_to_set04(base, src_csv, out_dir.name, quality, position)
    except KeyboardInterrupt:
        print("\nОтменено пользователем.")
        return None
    where = "Normal_mod" if quality == "normal" else f"Deformed_mod/{position}"
    print(f"\nСохранено в datasets/set_04/{where}: {dst}")

    truth = _truth_label_from_decision(quality, position)
    _print_and_save_session_metrics(
        out_dir,
        truth,
        kalman_log,
        rf_log,
        rf_smooth_log,
        tree_log,
        tree_smooth_log,
    )
    return (quality, position)


# ---------------------------------------------------------------------------
# Утилита: одно извлечение признаков → два предсказания
# ---------------------------------------------------------------------------


def _predict_both(
    feats: dict,
    rf_clf,
    rf_features: list[str],
    rf_labels: list[str],
    tree_clf,
    tree_features: list[str],
    tree_labels: list[str],
) -> tuple[tuple[str, float], tuple[str, float]]:
    rf_x = np.array([[feats.get(n, 0.0) for n in rf_features]], dtype=float)
    rf_proba = rf_clf.predict_proba(rf_x)[0]
    rf_i = int(np.argmax(rf_proba))

    tr_x = np.array([[feats.get(n, 0.0) for n in tree_features]], dtype=float)
    tr_proba = tree_clf.predict_proba(tr_x)[0]
    tr_i = int(np.argmax(tr_proba))
    return (rf_labels[rf_i], float(rf_proba[rf_i])), (tree_labels[tr_i], float(tr_proba[tr_i]))


# ---------------------------------------------------------------------------
# Live и offline режимы
# ---------------------------------------------------------------------------


def run_live(base: Path, calib_base: Path, rf_path: Path, tree_path: Path) -> Path:
    from DataPacketParser import DataPacketParser, XsDataPacket
    from SerialHandler import SerialHandler
    from XbusPacket import XbusPacket

    normal_dir = calib_base / "Normal_mod"
    if not normal_dir.is_dir():
        print(f"Нет {normal_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"RF модель:   {rf_path}")
    print(f"Tree модель: {tree_path}")
    print(f"Калибровка Калмана по: {normal_dir}")
    rf_bundle = load_model(rf_path)
    tree_bundle = load_model(tree_path)
    rf = RealtimeFaultDetector(rf_bundle)
    tree_clf = tree_bundle["classifier"]
    tree_features = list(tree_bundle["feature_names"])
    tree_labels = list(tree_bundle["class_labels"])
    if tree_labels != list(rf.class_labels):
        print(
            f"[!] Внимание: классы у RF и Tree различаются:\n  RF:   {rf.class_labels}\n  Tree: {tree_labels}",
            file=sys.stderr,
        )

    mu, R = calibrate_from_normal(normal_dir)
    kalm = KalmanFaultDetector(mu, R, q_factor=Q_FACTOR, ewma_alpha=EWMA_ALPHA)

    vib_rows: list[tuple[float, float, float, float, float]] = []
    kalman_log: list[tuple[float, float, int]] = []
    rf_log: list[tuple[float, str, float]] = []
    tree_log: list[tuple[float, str, float]] = []
    rf_smooth_buf: deque = deque(maxlen=SMOOTH_WIN)
    tree_smooth_buf: deque = deque(maxlen=SMOOTH_WIN)
    rf_smooth_log: list[tuple[float, str, float]] = []
    tree_smooth_log: list[tuple[float, str, float]] = []

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
    rf.start_time = time.time()

    def on_packet(raw_packet):
        xbus = XsDataPacket()
        DataPacketParser.parse_data_packet(raw_packet, xbus)
        if not xbus.accAvailable:
            return
        n_before = len(rf.vib_total)
        rf.add_accel(xbus.acc[0], xbus.acc[1], xbus.acc[2])
        if len(rf.vib_total) > n_before:
            t_rel = time.time() - rf.start_time
            total = float(rf.vib_total[-1])
            rx = float(rf.vib_x[-1])
            ry = float(rf.vib_y[-1])
            rz = float(rf.vib_z[-1])
            vib_rows.append((t_rel, total, rx, ry, rz))
            out = kalm.update(np.array([total, rx, ry, rz], dtype=float))
            kalman_log.append((t_rel, float(out["p_fault"]), int(out["p_fault"] >= 0.5)))

        step_counter[0] += 1
        if step_counter[0] % STEP != 0:
            return
        if len(rf.vib_total) < rf.window_size:
            return

        feats = extract_features(
            np.fromiter(rf.vib_total, dtype=float, count=rf.window_size),
            np.fromiter(rf.vib_x, dtype=float, count=rf.window_size),
            np.fromiter(rf.vib_y, dtype=float, count=rf.window_size),
            np.fromiter(rf.vib_z, dtype=float, count=rf.window_size),
        )
        (rf_lbl, rf_conf), (tr_lbl, tr_conf) = _predict_both(
            feats, rf.clf, rf.feature_names, rf.class_labels,
            tree_clf, tree_features, tree_labels,
        )
        t_s = time.time() - rf.start_time
        rf_log.append((t_s, rf_lbl, rf_conf))
        tree_log.append((t_s, tr_lbl, tr_conf))
        rf_smooth_buf.append((rf_lbl, rf_conf))
        tree_smooth_buf.append((tr_lbl, tr_conf))
        sm = _smoothed(rf_smooth_buf)
        if sm is not None:
            rf_smooth_log.append((t_s, sm[0], sm[1]))
        sm = _smoothed(tree_smooth_buf)
        if sm is not None:
            tree_smooth_log.append((t_s, sm[0], sm[1]))

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
        base, f"triple_live_{ts}",
        vib_rows, kalman_log,
        rf_log, rf_smooth_log,
        tree_log, tree_smooth_log,
        rf.class_labels,
    )
    print(f"Артефакты: {out_dir}")
    _maybe_export(
        base, out_dir, rf_smooth_log, tree_smooth_log, kalman_log, rf_log, tree_log,
    )
    return out_dir


def run_offline(
    base: Path,
    calib_base: Path,
    csv_path: Path,
    rf_path: Path,
    tree_path: Path,
) -> Path:
    normal_dir = calib_base / "Normal_mod"
    print(f"RF модель:   {rf_path}")
    print(f"Tree модель: {tree_path}")
    print(f"Калибровка Калмана по: {normal_dir}")
    rf_bundle = load_model(rf_path)
    tree_bundle = load_model(tree_path)
    rf = RealtimeFaultDetector(rf_bundle)
    tree_clf = tree_bundle["classifier"]
    tree_features = list(tree_bundle["feature_names"])
    tree_labels = list(tree_bundle["class_labels"])
    if tree_labels != list(rf.class_labels):
        print(
            f"[!] Внимание: классы у RF и Tree различаются:\n  RF:   {rf.class_labels}\n  Tree: {tree_labels}",
            file=sys.stderr,
        )

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
    rf_log: list[tuple[float, str, float]] = []
    tree_log: list[tuple[float, str, float]] = []
    rf_smooth_buf: deque = deque(maxlen=SMOOTH_WIN)
    tree_smooth_buf: deque = deque(maxlen=SMOOTH_WIN)
    rf_smooth_log: list[tuple[float, str, float]] = []
    tree_smooth_log: list[tuple[float, str, float]] = []

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
        (rf_lbl, rf_conf), (tr_lbl, tr_conf) = _predict_both(
            feats, rf.clf, rf.feature_names, rf.class_labels,
            tree_clf, tree_features, tree_labels,
        )
        rf_log.append((t_wall, rf_lbl, rf_conf))
        tree_log.append((t_wall, tr_lbl, tr_conf))
        rf_smooth_buf.append((rf_lbl, rf_conf))
        tree_smooth_buf.append((tr_lbl, tr_conf))
        sm = _smoothed(rf_smooth_buf)
        if sm is not None:
            rf_smooth_log.append((t_wall, sm[0], sm[1]))
        sm = _smoothed(tree_smooth_buf)
        if sm is not None:
            tree_smooth_log.append((t_wall, sm[0], sm[1]))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _save_artifacts(
        base, f"triple_offline_{ts}",
        vib_rows, kalman_log,
        rf_log, rf_smooth_log,
        tree_log, tree_smooth_log,
        rf.class_labels,
    )
    print(f"Артефакты: {out_dir}")
    _maybe_export(
        base, out_dir, rf_smooth_log, tree_smooth_log, kalman_log, rf_log, tree_log,
    )
    return out_dir


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Калман + RF + Decision Tree, один поток, общий график и экспорт в set_04",
    )
    parser.add_argument(
        "--base", type=Path, default=here,
        help="Корень проекта (модель, detection_logs, datasets/set_04)",
    )
    parser.add_argument(
        "--calib", type=Path, default=here / "datasets" / "set_03",
        help="Датасет с Normal_mod для калибровки Калмана",
    )
    parser.add_argument(
        "--offline", type=Path, default=None,
        help="Режим без Xsens: vibration_log.csv (как в main.py)",
    )
    parser.add_argument(
        "--rf-model", type=Path, default=None,
        help=f"pkl с Random Forest (по умолчанию ищется: {', '.join(DEFAULT_RF_CANDIDATES)})",
    )
    parser.add_argument(
        "--tree-model", type=Path, default=None,
        help=f"pkl с Decision Tree (по умолчанию ищется: {', '.join(DEFAULT_TREE_CANDIDATES)})",
    )
    args = parser.parse_args()

    base = args.base.resolve()
    rf_path = _resolve_model(base, args.rf_model, DEFAULT_RF_CANDIDATES)
    tree_path = _resolve_model(base, args.tree_model, DEFAULT_TREE_CANDIDATES)

    if args.offline is not None:
        run_offline(base, args.calib.resolve(), args.offline.resolve(), rf_path, tree_path)
    else:
        run_live(base, args.calib.resolve(), rf_path, tree_path)


if __name__ == "__main__":
    main()
