#!/usr/bin/env python3
"""
Один процесс: общий поток Xsens → фильтр Калмана (бинарный FAULT) + Random Forest
(мультикласс). Во время работы в консоль ничего не пишется.

По Ctrl+C (или конец файла в --offline): сохраняется график сравнения в
detection_logs/dual_compare_<timestamp>/compare.png (+ CSV).

Использование:
  python3 dual_kalman_rf.py
  python3 dual_kalman_rf.py --calib datasets/set_03
  python3 dual_kalman_rf.py --offline path/to/vibration_log.csv
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
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from DataPacketParser import DataPacketParser, XsDataPacket
from SerialHandler import SerialHandler
from XbusPacket import XbusPacket
from feature_extraction import extract_features

from kalman_fault_detector import (
    KalmanFaultDetector,
    calibrate_from_normal,
    EWMA_ALPHA,
    Q_FACTOR,
)
from realtime_detector import (
    RealtimeFaultDetector,
    STEP,
    WINDOW_SIZE,
    load_model,
)


def _label_to_idx(labels: list[str], name: str) -> int:
    try:
        return labels.index(name)
    except ValueError:
        return -1


def plot_compare(
    class_labels: list[str],
    kalman_t: np.ndarray,
    kalman_p: np.ndarray,
    rf_t: np.ndarray,
    rf_label_idx: np.ndarray,
    rf_smooth_t: np.ndarray | None,
    rf_smooth_idx: np.ndarray | None,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    ax1, ax2, ax3 = axes
    ax1.plot(kalman_t, kalman_p, color="tab:blue", linewidth=1.2, label="Калман p_fault")
    ax1.axhline(0.5, color="tab:red", linestyle="--", linewidth=1, label="порог 0.5")
    ax1.set_ylabel("Калман\np(FAULT)")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Калман vs Random Forest — одна сессия, один поток данных")

    ax2.step(rf_t, rf_label_idx, where="post", color="tab:green", linewidth=1.0, label="RF (сырое)")
    if rf_smooth_t is not None and len(rf_smooth_t) > 0:
        ax2.step(
            rf_smooth_t,
            rf_smooth_idx,
            where="post",
            color="tab:orange",
            linewidth=1.2,
            alpha=0.85,
            label="RF (сглаж.)",
        )
    ax2.set_yticks(range(len(class_labels)))
    ax2.set_yticklabels(class_labels, fontsize=8)
    ax2.set_ylabel("RF класс")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    k_bin = (kalman_p >= 0.5).astype(float)
    ax3.step(kalman_t, k_bin, where="post", color="tab:blue", linewidth=1.2, label="Калман FAULT")
    if rf_smooth_t is not None and len(rf_smooth_t) > 0:
        rf_def = (rf_smooth_idx > 0).astype(float)
        ax3.step(
            rf_smooth_t,
            rf_def,
            where="post",
            color="tab:orange",
            linewidth=1.2,
            label="RF поломка (сглаж., ≠ normal)",
        )
    else:
        rf_def = (rf_label_idx > 0).astype(float)
        ax3.step(rf_t, rf_def, where="post", color="tab:green", linewidth=1.0, label="RF поломка (сырое)")
    ax3.set_ylabel("бинарно\n0=норма 1=поломка")
    ax3.set_xlabel("время, с")
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if os.environ.get("DISPLAY") or sys.platform == "darwin":
        plt.show()
    plt.close(fig)


def run_live(base: Path, calib_base: Path) -> Path:
    model_path = base / "propeller_fault_model.pkl"
    if not model_path.exists():
        print(f"Нет модели: {model_path}", file=sys.stderr)
        sys.exit(1)

    normal_dir = calib_base / "Normal_mod"
    if not normal_dir.is_dir():
        print(f"Нет {normal_dir}", file=sys.stderr)
        sys.exit(1)

    mu, R = calibrate_from_normal(normal_dir)
    bundle = load_model(model_path)
    rf = RealtimeFaultDetector(bundle)
    kalm = KalmanFaultDetector(mu, R, q_factor=Q_FACTOR, ewma_alpha=EWMA_ALPHA)

    kalman_log: list[tuple[float, float, int]] = []
    smooth_log: list[tuple[float, str, float]] = []

    serial = SerialHandler("/dev/ttyUSB0", 115200)
    go_to_config = bytes.fromhex("FA FF 30 00")
    go_to_measurement = bytes.fromhex("FA FF 10 00")
    serial.send_with_checksum(go_to_config)
    time.sleep(0.1)
    config_mti30 = bytes.fromhex(
        "FA FF C0 20 10 20 FF FF 10 60 FF FF "
        "20 30 00 64 40 20 00 64 40 30 00 64 "
        "80 20 00 64 C0 20 00 64 E0 20 FF FF"
    )
    serial.send_with_checksum(config_mti30)
    time.sleep(0.1)
    serial.send_with_checksum(go_to_measurement)

    step_counter = [0]

    def on_packet(raw_packet):
        xbus = XsDataPacket()
        DataPacketParser.parse_data_packet(raw_packet, xbus)
        if not xbus.accAvailable:
            return
        n_before = len(rf.vibration_log)
        rf.add_accel(xbus.acc[0], xbus.acc[1], xbus.acc[2])
        if len(rf.vibration_log) > n_before:
            t_rel, total, rx, ry, rz = rf.vibration_log[-1]
            out = kalm.update(np.array([total, rx, ry, rz], dtype=float))
            kalman_log.append((t_rel, float(out["p_fault"]), int(out["p_fault"] >= 0.5)))

        step_counter[0] += 1
        if step_counter[0] % STEP != 0:
            return
        if rf.predict() is None:
            return
        sm = rf.smoothed_prediction()
        if sm is not None:
            t_s = rf.prediction_log[-1][0]
            smooth_log.append((t_s, sm[0], sm[1]))

    packet = XbusPacket(on_data_available=on_packet)
    try:
        while True:
            b = serial.read_byte()
            if b:
                packet.feed_byte(b)
    except KeyboardInterrupt:
        pass

    return _save_and_plot(base, rf, kalman_log, smooth_log)


def run_offline(base: Path, calib_base: Path, csv_path: Path) -> Path:
    model_path = base / "propeller_fault_model.pkl"
    if not model_path.exists():
        print(f"Нет модели: {model_path}", file=sys.stderr)
        sys.exit(1)
    normal_dir = calib_base / "Normal_mod"
    mu, R = calibrate_from_normal(normal_dir)
    bundle = load_model(model_path)
    rf = RealtimeFaultDetector(bundle)
    kalm = KalmanFaultDetector(mu, R, q_factor=Q_FACTOR, ewma_alpha=EWMA_ALPHA)

    df = pd.read_csv(csv_path)
    cols = ["total_vibration", "rms_x", "rms_y", "rms_z"]
    if not set(cols).issubset(df.columns):
        print(f"В CSV нужны колонки: {cols}", file=sys.stderr)
        sys.exit(1)

    kalman_log: list[tuple[float, float, int]] = []
    smooth_log: list[tuple[float, str, float]] = []

    times = (
        df["time_seconds"].to_numpy(dtype=float)
        if "time_seconds" in df.columns
        else np.arange(len(df), dtype=float) * 0.01
    )

    vib_total: deque = deque(maxlen=WINDOW_SIZE)
    vib_x: deque = deque(maxlen=WINDOW_SIZE)
    vib_y: deque = deque(maxlen=WINDOW_SIZE)
    vib_z: deque = deque(maxlen=WINDOW_SIZE)
    step_i = 0

    for row_idx, (_, row) in enumerate(df.iterrows()):
        z = row[cols].to_numpy(dtype=float)
        t_wall = float(times[row_idx]) if row_idx < len(times) else row_idx * 0.01
        vib_total.append(z[0])
        vib_x.append(z[1])
        vib_y.append(z[2])
        vib_z.append(z[3])

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
        x_vec = np.array([[feats.get(n, 0.0) for n in rf.feature_names]], dtype=float)
        proba = rf.clf.predict_proba(x_vec)[0]
        best = int(np.argmax(proba))
        lbl = rf.class_labels[best]
        conf = float(proba[best])
        rf.smoothing_buf.append((lbl, conf))
        rf.prediction_log.append((t_wall, lbl, conf))
        sm = rf.smoothed_prediction()
        if sm is not None:
            smooth_log.append((t_wall, sm[0], sm[1]))

    out_dir = base / "detection_logs" / f"dual_offline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csvs(out_dir, kalman_log, rf.prediction_log, smooth_log, rf.class_labels)
    _do_plot(out_dir, rf.class_labels, kalman_log, rf.prediction_log, smooth_log)
    return out_dir


def _save_and_plot(
    base: Path,
    rf: RealtimeFaultDetector,
    kalman_log: list[tuple[float, float, int]],
    smooth_log: list[tuple[float, str, float]],
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / "detection_logs" / f"dual_compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csvs(out_dir, kalman_log, rf.prediction_log, smooth_log, rf.class_labels)
    _do_plot(out_dir, rf.class_labels, kalman_log, rf.prediction_log, smooth_log)
    return out_dir


def _write_csvs(
    out_dir: Path,
    kalman_log: list[tuple[float, float, int]],
    rf_log: list[tuple[float, str, float]],
    smooth_log: list[tuple[float, str, float]],
    class_labels: list[str],
) -> None:
    k_path = out_dir / "kalman_series.csv"
    with open(k_path, "w") as f:
        f.write("time_s,p_fault,fault_bin\n")
        for t, p, b in kalman_log:
            f.write(f"{t:.4f},{p:.6f},{b}\n")

    r_path = out_dir / "rf_series.csv"
    with open(r_path, "w") as f:
        f.write("time_s,label,confidence\n")
        for t, lbl, c in rf_log:
            f.write(f"{t:.4f},{lbl},{c:.6f}\n")

    s_path = out_dir / "rf_smoothed.csv"
    with open(s_path, "w") as f:
        f.write("time_s,label,confidence\n")
        for t, lbl, c in smooth_log:
            f.write(f"{t:.4f},{lbl},{c:.6f}\n")

    meta = out_dir / "README.txt"
    with open(meta, "w") as f:
        f.write("Классы RF: " + ", ".join(class_labels) + "\n")
        f.write("Калман: бинарный p_fault (FAULT если >= 0.5)\n")


def _do_plot(
    out_dir: Path,
    class_labels: list[str],
    kalman_log: list[tuple[float, float, int]],
    rf_log: list[tuple[float, str, float]],
    smooth_log: list[tuple[float, str, float]],
) -> None:
    if not kalman_log and not rf_log:
        return
    kt = np.array([x[0] for x in kalman_log]) if kalman_log else np.array([])
    kp = np.array([x[1] for x in kalman_log]) if kalman_log else np.array([])

    rf_t = np.array([x[0] for x in rf_log]) if rf_log else np.array([])
    rf_idx = (
        np.array([_label_to_idx(class_labels, x[1]) for x in rf_log])
        if rf_log
        else np.array([])
    )

    st = np.array([x[0] for x in smooth_log]) if smooth_log else None
    sidx = (
        np.array([_label_to_idx(class_labels, x[1]) for x in smooth_log])
        if smooth_log
        else None
    )

    png = out_dir / "compare.png"
    plot_compare(class_labels, kt, kp, rf_t, rf_idx, st, sidx, png)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Калман + RF, один поток, график в конце")
    parser.add_argument(
        "--base",
        type=Path,
        default=here,
        help="Корень проекта (модель, detection_logs)",
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=here / "datasets" / "set_03",
        help="Датасет с Normal_mod для калибровки Калмана",
    )
    parser.add_argument(
        "--offline",
        type=Path,
        default=None,
        help="Режим без Xsens: vibration_log.csv (колонки как в main.py)",
    )
    args = parser.parse_args()

    if args.offline is not None:
        run_offline(args.base.resolve(), args.calib.resolve(), args.offline.resolve())
    else:
        run_live(args.base.resolve(), args.calib.resolve())


if __name__ == "__main__":
    main()
