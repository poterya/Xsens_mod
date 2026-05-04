#!/usr/bin/env python3
"""
Один процесс: общий поток Xsens → ТРИ бинарных метода детекции одновременно:
  1. Калман-фильтр (бинарный FAULT по Mahalanobis);
  2. Random Forest, бинарный (normal vs fault);
  3. Decision Tree, бинарный.

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
"""
from __future__ import annotations

import os

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import argparse
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
    STEP,
    WINDOW_SIZE,
    load_model,
)

from dual_kalman_rf import (
    SMOOTH_WIN,
    _ask_quality_and_position,
    _export_to_set04,
    _has_gui_display,
    _resolve_model,
    _smoothed,
    _truth_label_from_decision,
)

OUTPUT_ROOT_NAME = "binary_detection_logs"

DEFAULT_RF_BIN_CANDIDATES = (
    "propeller_fault_rf_binary_set03.pkl",
    "propeller_fault_rf_binary.pkl",
)
DEFAULT_TREE_BIN_CANDIDATES = (
    "propeller_fault_tree_binary_set03.pkl",
    "propeller_fault_tree_binary.pkl",
)
BINARY_LABELS = ("normal", "fault")


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
    from DataPacketParser import DataPacketParser, XsDataPacket
    from SerialHandler import SerialHandler
    from XbusPacket import XbusPacket

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
    here = Path(__file__).resolve().parent
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
    args = parser.parse_args()

    base = args.base.resolve()
    rf_bin_path = _resolve_model(base, args.rf_binary_model, DEFAULT_RF_BIN_CANDIDATES)
    tree_bin_path = _resolve_model(base, args.tree_binary_model, DEFAULT_TREE_BIN_CANDIDATES)

    if args.offline is not None:
        run_offline(
            base, args.calib.resolve(), args.offline.resolve(),
            rf_bin_path, tree_bin_path,
        )
    else:
        run_live(base, args.calib.resolve(), rf_bin_path, tree_bin_path)


if __name__ == "__main__":
    main()
