#!/usr/bin/env python3
"""Run NN v2.0 detectors (forest + tree) together with Kalman.

Methods:
  1. Kalman filter:                binary FAULT probability;
  2. NN v2.0 forest binary:        normal vs fault (small MLP ensemble);
  3. NN v2.0 tree binary:          normal vs fault (single compact MLP);
  4. NN v2.0 forest position:      normal + 4 propeller positions;
  5. NN v2.0 tree position:        normal + 4 propeller positions.

Artifacts are saved to nn_v2_detection_logs/nn_v2_<live|offline>_<ts>/.
After the user labels the flight (n/d + position), vibration_log.csv is copied
to datasets/set_03 in the matching folder, and per-model accuracy is printed.
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
    EWMA_ALPHA,
    Q_FACTOR,
    KalmanFaultDetector,
    calibrate_from_normal,
)
from realtime_detector import STEP, WINDOW_SIZE, load_model

from all_detectors import _RMSWindow, _binary_proba_fault
from dual_kalman_rf import (
    SMOOTH_WIN,
    _ask_quality_and_position,
    _export_to_dataset,
    _has_gui_display,
    _resolve_model,
    _smoothed,
    _truth_label_from_decision,
)
import nn_forest  # noqa: F401  (so MLPForest is importable for pickle.load)

OUTPUT_ROOT_NAME = "nn_v2_detection_logs"
EXPORT_DATASET = "set_03"

DEFAULT_FOREST_BINARY_CANDIDATES = (
    "propeller_fault_nn_v2_forest_binary_set05.pkl",
)
DEFAULT_TREE_BINARY_CANDIDATES = (
    "propeller_fault_nn_v2_tree_binary_set05.pkl",
)
DEFAULT_FOREST_POSITION_CANDIDATES = (
    "propeller_fault_nn_v2_forest_position_set05.pkl",
)
DEFAULT_TREE_POSITION_CANDIDATES = (
    "propeller_fault_nn_v2_tree_position_set05.pkl",
)


def _label_to_idx(labels: list[str], label: str) -> int:
    try:
        return labels.index(label)
    except ValueError:
        return -1


def _predict_position(bundle: dict, feats: dict) -> tuple[str, float]:
    feature_names = list(bundle["feature_names"])
    labels = list(bundle["class_labels"])
    clf = bundle["classifier"]
    x = np.array([[feats.get(name, 0.0) for name in feature_names]], dtype=float)
    proba = clf.predict_proba(x)[0]
    best = int(np.argmax(proba))
    return labels[best], float(proba[best])


def _predict_binary(bundle: dict, feats: dict) -> tuple[str, float, float]:
    feature_names = list(bundle["feature_names"])
    labels = list(bundle["class_labels"])
    clf = bundle["classifier"]
    x = np.array([[feats.get(name, 0.0) for name in feature_names]], dtype=float)
    return _binary_proba_fault(clf, x, labels)


def _write_series(out_dir: Path, name: str, header: str, rows) -> None:
    with open(out_dir / name, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def _write_vibration_log(out_dir: Path, vib_rows) -> Path:
    path = out_dir / "vibration_log.csv"
    with open(path, "w") as f:
        f.write("time_seconds,total_vibration,rms_x,rms_y,rms_z\n")
        for t, total, rx, ry, rz in vib_rows:
            f.write(f"{t:.3f},{total:.6f},{rx:.6f},{ry:.6f},{rz:.6f}\n")
    return path


def _write_logs(
    out_dir: Path,
    kalman_log,
    forest_bin_log,
    forest_bin_smooth_log,
    tree_bin_log,
    tree_bin_smooth_log,
    forest_pos_log,
    forest_pos_smooth_log,
    tree_pos_log,
    tree_pos_smooth_log,
    position_labels: list[str],
) -> None:
    _write_series(
        out_dir,
        "kalman_series.csv",
        "time_s,p_fault,fault_bin",
        ((f"{t:.4f}", f"{p:.6f}", b) for t, p, b in kalman_log),
    )
    for name, log in (
        ("nn_forest_binary_series.csv", forest_bin_log),
        ("nn_tree_binary_series.csv", tree_bin_log),
    ):
        _write_series(
            out_dir,
            name,
            "time_s,label,confidence,p_fault",
            ((f"{t:.4f}", lbl, f"{c:.6f}", f"{pf:.6f}") for t, lbl, c, pf in log),
        )
    for name, log in (
        ("nn_forest_binary_smoothed.csv", forest_bin_smooth_log),
        ("nn_tree_binary_smoothed.csv", tree_bin_smooth_log),
        ("nn_forest_position_series.csv", forest_pos_log),
        ("nn_tree_position_series.csv", tree_pos_log),
        ("nn_forest_position_smoothed.csv", forest_pos_smooth_log),
        ("nn_tree_position_smoothed.csv", tree_pos_smooth_log),
    ):
        _write_series(
            out_dir,
            name,
            "time_s,label,confidence",
            ((f"{t:.4f}", lbl, f"{c:.6f}") for t, lbl, c in log),
        )
    with open(out_dir / "README.txt", "w") as f:
        f.write("Kalman:                 binary p_fault (FAULT if >= 0.5)\n")
        f.write("NN forest binary:       normal, fault (MLP ensemble)\n")
        f.write("NN tree binary:         normal, fault (single MLP)\n")
        f.write("NN forest position:     " + ", ".join(position_labels) + " (ensemble)\n")
        f.write("NN tree position:       " + ", ".join(position_labels) + " (single MLP)\n")
        f.write("compare.png saved to disk even without DISPLAY\n")
        f.write("metrics_after_label.txt appears after n/d console answer\n")


def _plot(
    out_dir: Path,
    position_labels: list[str],
    kalman_log,
    forest_bin_log,
    forest_bin_smooth_log,
    tree_bin_log,
    tree_bin_smooth_log,
    forest_pos_log,
    forest_pos_smooth_log,
    tree_pos_log,
    tree_pos_smooth_log,
) -> None:
    if not (kalman_log or forest_bin_log or tree_bin_log or forest_pos_log or tree_pos_log):
        return

    def arr(log, *cols):
        if not log:
            return tuple(np.array([]) for _ in cols)
        return tuple(np.array([row[col] for row in log]) for col in cols)

    kt, kp = arr(kalman_log, 0, 1)
    fbt, fbp = arr(forest_bin_log, 0, 3)
    tbt, tbp = arr(tree_bin_log, 0, 3)

    def label_array(log, labels: list[str]):
        if not log:
            return np.array([]), np.array([])
        return (
            np.array([row[0] for row in log]),
            np.array([_label_to_idx(labels, row[1]) for row in log]),
        )

    fpos_t, fpos_idx = label_array(forest_pos_log, position_labels)
    fpos_st, fpos_sidx = label_array(forest_pos_smooth_log, position_labels)
    tpos_t, tpos_idx = label_array(tree_pos_log, position_labels)
    tpos_st, tpos_sidx = label_array(tree_pos_smooth_log, position_labels)

    def bin_label_array(log):
        if not log:
            return np.array([]), np.array([])
        return (
            np.array([row[0] for row in log]),
            np.array([0 if row[1] == "normal" else 1 for row in log]),
        )

    fbin_st, fbin_sidx = bin_label_array(forest_bin_smooth_log or forest_bin_log)
    tbin_st, tbin_sidx = bin_label_array(tree_bin_smooth_log or tree_bin_log)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    if len(kt):
        ax1.plot(kt, kp, color="tab:blue", linewidth=1.2, label="Kalman p_fault")
    if len(fbt):
        ax1.plot(fbt, fbp, color="tab:red", linewidth=1.0, label="NN forest p_fault")
    if len(tbt):
        ax1.plot(tbt, tbp, color="tab:purple", linewidth=1.0, label="NN tree p_fault")
    ax1.axhline(0.5, color="0.4", linestyle="--", linewidth=1, label="threshold 0.5")
    ax1.set_ylabel("p(FAULT)")
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("NN v2.0 forest + tree + Kalman")

    if len(fpos_t):
        ax2.step(fpos_t, fpos_idx, where="post", color="tab:green", linewidth=0.9, alpha=0.6, label="forest raw")
    if len(fpos_st):
        ax2.step(fpos_st, fpos_sidx, where="post", color="tab:orange", linewidth=1.4, label="forest smooth")
    ax2.set_yticks(range(len(position_labels)))
    ax2.set_yticklabels(position_labels, fontsize=8)
    ax2.set_ylabel("forest position")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    if len(tpos_t):
        ax3.step(tpos_t, tpos_idx, where="post", color="tab:cyan", linewidth=0.9, alpha=0.6, label="tree raw")
    if len(tpos_st):
        ax3.step(tpos_st, tpos_sidx, where="post", color="tab:brown", linewidth=1.4, label="tree smooth")
    ax3.set_yticks(range(len(position_labels)))
    ax3.set_yticklabels(position_labels, fontsize=8)
    ax3.set_ylabel("tree position")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    if len(kt):
        ax4.step(kt, (kp >= 0.5).astype(float), where="post", color="tab:blue", linewidth=1.4, label="Kalman FAULT")
    if len(fbin_st):
        ax4.step(fbin_st, fbin_sidx, where="post", color="tab:red", linewidth=1.0, label="forest binary")
    if len(tbin_st):
        ax4.step(tbin_st, tbin_sidx, where="post", color="tab:purple", linewidth=1.0, label="tree binary")
    if len(fpos_st):
        ax4.step(
            fpos_st,
            (fpos_sidx > 0).astype(float),
            where="post",
            color="tab:orange",
            linewidth=0.9,
            alpha=0.7,
            label="forest pos != normal",
        )
    if len(tpos_st):
        ax4.step(
            tpos_st,
            (tpos_sidx > 0).astype(float),
            where="post",
            color="tab:brown",
            linewidth=0.9,
            alpha=0.7,
            label="tree pos != normal",
        )
    ax4.set_ylabel("binary\n0=normal 1=fault")
    ax4.set_xlabel("time, s")
    ax4.set_ylim(-0.1, 1.15)
    ax4.legend(loc="upper right", fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "compare.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if not path.is_file() or path.stat().st_size < 100:
        print(f"[!] Не удалось сохранить график: {path}", file=sys.stderr)
    else:
        print(f"График сохранён: {path.resolve()}")
    if _has_gui_display():
        try:
            plt.show(block=False)
            plt.pause(0.05)
        except Exception:
            pass
    plt.close(fig)


def _save_artifacts(
    base: Path,
    name: str,
    vib_rows,
    kalman_log,
    forest_bin_log,
    forest_bin_smooth_log,
    tree_bin_log,
    tree_bin_smooth_log,
    forest_pos_log,
    forest_pos_smooth_log,
    tree_pos_log,
    tree_pos_smooth_log,
    position_labels: list[str],
) -> Path:
    out_dir = base / OUTPUT_ROOT_NAME / name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_logs(
        out_dir,
        kalman_log,
        forest_bin_log,
        forest_bin_smooth_log,
        tree_bin_log,
        tree_bin_smooth_log,
        forest_pos_log,
        forest_pos_smooth_log,
        tree_pos_log,
        tree_pos_smooth_log,
        position_labels,
    )
    _write_vibration_log(out_dir, vib_rows)
    _plot(
        out_dir,
        position_labels,
        kalman_log,
        forest_bin_log,
        forest_bin_smooth_log,
        tree_bin_log,
        tree_bin_smooth_log,
        forest_pos_log,
        forest_pos_smooth_log,
        tree_pos_log,
        tree_pos_smooth_log,
    )
    return out_dir


def _acc_binary(log, truth_fault: bool):
    if not log:
        return 0, 0, None
    want = "fault" if truth_fault else "normal"
    ok = sum(1 for row in log if row[1] == want)
    return ok, len(log), ok / len(log)


def _acc_position(log, truth_label: str):
    if not log:
        return 0, 0, None
    ok = sum(1 for row in log if row[1] == truth_label)
    return ok, len(log), ok / len(log)


def _acc_kalman(kalman_log, truth_fault: bool):
    if not kalman_log:
        return 0, 0, None
    want = 1 if truth_fault else 0
    ok = sum(1 for row in kalman_log if row[2] == want)
    return ok, len(kalman_log), ok / len(kalman_log)


def _fmt(name: str, values) -> str:
    ok, total, acc = values
    return f"{name}: {ok}/{total}" + (f" = {100 * acc:.1f}%" if acc is not None else " (нет точек)")


def _hint(forest_bin_smooth, tree_bin_smooth, forest_pos_smooth, tree_pos_smooth, kalman_log) -> str:
    def top(log):
        labels = [row[1] for row in log]
        return Counter(labels).most_common(1)[0][0] if labels else "-"

    kal = "нет"
    if kalman_log:
        kal = f"FAULT доля {sum(row[2] for row in kalman_log) / len(kalman_log):.2f}"
    return (
        f"forest binary «{top(forest_bin_smooth)}», "
        f"tree binary «{top(tree_bin_smooth)}», "
        f"forest position «{top(forest_pos_smooth)}», "
        f"tree position «{top(tree_pos_smooth)}»; Калман: {kal}"
    )


def _maybe_export_and_metrics(
    base: Path,
    out_dir: Path,
    kalman_log,
    forest_bin_log,
    forest_bin_smooth_log,
    tree_bin_log,
    tree_bin_smooth_log,
    forest_pos_log,
    forest_pos_smooth_log,
    tree_pos_log,
    tree_pos_smooth_log,
) -> None:
    src_csv = out_dir / "vibration_log.csv"
    decision = _ask_quality_and_position(
        default_hint=_hint(
            forest_bin_smooth_log,
            tree_bin_smooth_log,
            forest_pos_smooth_log,
            tree_pos_smooth_log,
            kalman_log,
        ),
        dataset_name=EXPORT_DATASET,
    )
    if decision is None:
        print(f"Экспорт в {EXPORT_DATASET} пропущен.")
        return
    quality, position = decision
    dst = _export_to_dataset(
        base, src_csv, out_dir.name, quality, position, EXPORT_DATASET
    )
    where = "Normal_mod" if quality == "normal" else f"Deformed_mod/{position}"
    print(f"\nСохранено в datasets/{EXPORT_DATASET}/{where}: {dst}")

    truth_label = _truth_label_from_decision(quality, position)
    truth_fault = truth_label != "normal"
    lines = [
        f"Истина (ваша разметка): {truth_label}",
        "",
        _fmt("Калман (бинарно)        ", _acc_kalman(kalman_log, truth_fault)),
        _fmt("NN forest binary, сырой ", _acc_binary(forest_bin_log, truth_fault)),
        _fmt("NN forest binary, сглаж.", _acc_binary(forest_bin_smooth_log, truth_fault)),
        _fmt("NN tree   binary, сырой ", _acc_binary(tree_bin_log, truth_fault)),
        _fmt("NN tree   binary, сглаж.", _acc_binary(tree_bin_smooth_log, truth_fault)),
        _fmt("NN forest pos,    сырой ", _acc_position(forest_pos_log, truth_label)),
        _fmt("NN forest pos,    сглаж.", _acc_position(forest_pos_smooth_log, truth_label)),
        _fmt("NN tree   pos,    сырой ", _acc_position(tree_pos_log, truth_label)),
        _fmt("NN tree   pos,    сглаж.", _acc_position(tree_pos_smooth_log, truth_label)),
    ]
    print("\n=== Точность по вашей разметке ===")
    for line in lines:
        print(line)
    metrics = out_dir / "metrics_after_label.txt"
    metrics.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nМетрики записаны: {metrics.resolve()}")


def _process_window(
    feats: dict,
    forest_bin_bundle: dict,
    tree_bin_bundle: dict,
    forest_pos_bundle: dict,
    tree_pos_bundle: dict,
):
    return (
        _predict_binary(forest_bin_bundle, feats),
        _predict_binary(tree_bin_bundle, feats),
        _predict_position(forest_pos_bundle, feats),
        _predict_position(tree_pos_bundle, feats),
    )


def _append_smoothed(buf: deque, dest: list, t: float, pred: tuple) -> None:
    buf.append((pred[0], pred[1]))
    sm = _smoothed(buf)
    if sm is not None:
        dest.append((t, sm[0], sm[1]))


def _run_common_offline(
    base: Path,
    calib_base: Path,
    csv_path: Path,
    forest_bin_bundle: dict,
    tree_bin_bundle: dict,
    forest_pos_bundle: dict,
    tree_pos_bundle: dict,
) -> Path:
    normal_dir = calib_base / "Normal_mod"
    mu, R = calibrate_from_normal(normal_dir)
    kalm = KalmanFaultDetector(mu, R, q_factor=Q_FACTOR, ewma_alpha=EWMA_ALPHA)

    df = pd.read_csv(csv_path)
    cols = ["total_vibration", "rms_x", "rms_y", "rms_z"]
    if not set(cols).issubset(df.columns):
        raise SystemExit(f"В CSV нужны колонки: {cols}")
    times = df["time_seconds"].to_numpy(dtype=float) if "time_seconds" in df.columns else np.arange(len(df)) * 0.01

    vib_total: deque = deque(maxlen=WINDOW_SIZE)
    vib_x: deque = deque(maxlen=WINDOW_SIZE)
    vib_y: deque = deque(maxlen=WINDOW_SIZE)
    vib_z: deque = deque(maxlen=WINDOW_SIZE)

    vib_rows = []
    kalman_log = []
    forest_bin_log = []
    tree_bin_log = []
    forest_pos_log = []
    tree_pos_log = []
    forest_bin_buf: deque = deque(maxlen=SMOOTH_WIN)
    tree_bin_buf: deque = deque(maxlen=SMOOTH_WIN)
    forest_pos_buf: deque = deque(maxlen=SMOOTH_WIN)
    tree_pos_buf: deque = deque(maxlen=SMOOTH_WIN)
    forest_bin_smooth_log = []
    tree_bin_smooth_log = []
    forest_pos_smooth_log = []
    tree_pos_smooth_log = []
    step_i = 0

    for row_idx, (_, row) in enumerate(df.iterrows()):
        z = row[cols].to_numpy(dtype=float)
        t = float(times[row_idx]) if row_idx < len(times) else row_idx * 0.01
        vib_total.append(z[0]); vib_x.append(z[1]); vib_y.append(z[2]); vib_z.append(z[3])
        vib_rows.append((t, float(z[0]), float(z[1]), float(z[2]), float(z[3])))
        out = kalm.update(z)
        kalman_log.append((t, float(out["p_fault"]), int(out["p_fault"] >= 0.5)))

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
        f_bin, t_bin, f_pos, t_pos = _process_window(
            feats, forest_bin_bundle, tree_bin_bundle, forest_pos_bundle, tree_pos_bundle
        )
        forest_bin_log.append((t, f_bin[0], f_bin[1], f_bin[2]))
        tree_bin_log.append((t, t_bin[0], t_bin[1], t_bin[2]))
        forest_pos_log.append((t, f_pos[0], f_pos[1]))
        tree_pos_log.append((t, t_pos[0], t_pos[1]))
        _append_smoothed(forest_bin_buf, forest_bin_smooth_log, t, f_bin)
        _append_smoothed(tree_bin_buf, tree_bin_smooth_log, t, t_bin)
        _append_smoothed(forest_pos_buf, forest_pos_smooth_log, t, f_pos)
        _append_smoothed(tree_pos_buf, tree_pos_smooth_log, t, t_pos)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    position_labels = list(forest_pos_bundle["class_labels"])
    out_dir = _save_artifacts(
        base,
        f"nn_v2_offline_{ts}",
        vib_rows,
        kalman_log,
        forest_bin_log,
        forest_bin_smooth_log,
        tree_bin_log,
        tree_bin_smooth_log,
        forest_pos_log,
        forest_pos_smooth_log,
        tree_pos_log,
        tree_pos_smooth_log,
        position_labels,
    )
    print(f"Артефакты: {out_dir}")
    _maybe_export_and_metrics(
        base,
        out_dir,
        kalman_log,
        forest_bin_log,
        forest_bin_smooth_log,
        tree_bin_log,
        tree_bin_smooth_log,
        forest_pos_log,
        forest_pos_smooth_log,
        tree_pos_log,
        tree_pos_smooth_log,
    )
    return out_dir


def run_live(
    base: Path,
    calib_base: Path,
    forest_bin_bundle: dict,
    tree_bin_bundle: dict,
    forest_pos_bundle: dict,
    tree_pos_bundle: dict,
) -> Path:
    from DataPacketParser import DataPacketParser, XsDataPacket
    from SerialHandler import SerialHandler
    from XbusPacket import XbusPacket

    normal_dir = calib_base / "Normal_mod"
    mu, R = calibrate_from_normal(normal_dir)
    kalm = KalmanFaultDetector(mu, R, q_factor=Q_FACTOR, ewma_alpha=EWMA_ALPHA)
    rmsw = _RMSWindow(WINDOW_SIZE)

    vib_rows = []
    kalman_log = []
    forest_bin_log = []
    tree_bin_log = []
    forest_pos_log = []
    tree_pos_log = []
    forest_bin_buf: deque = deque(maxlen=SMOOTH_WIN)
    tree_bin_buf: deque = deque(maxlen=SMOOTH_WIN)
    forest_pos_buf: deque = deque(maxlen=SMOOTH_WIN)
    tree_pos_buf: deque = deque(maxlen=SMOOTH_WIN)
    forest_bin_smooth_log = []
    tree_bin_smooth_log = []
    forest_pos_smooth_log = []
    tree_pos_smooth_log = []

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

    start_time = time.time()
    step_counter = [0]

    def on_packet(raw_packet):
        xbus = XsDataPacket()
        DataPacketParser.parse_data_packet(raw_packet, xbus)
        if not xbus.accAvailable:
            return
        n_before = len(rmsw.vib_total)
        rmsw.add(xbus.acc[0], xbus.acc[1], xbus.acc[2])
        if len(rmsw.vib_total) > n_before:
            t = time.time() - start_time
            total = float(rmsw.vib_total[-1])
            rx = float(rmsw.vib_x[-1])
            ry = float(rmsw.vib_y[-1])
            rz = float(rmsw.vib_z[-1])
            vib_rows.append((t, total, rx, ry, rz))
            out = kalm.update(np.array([total, rx, ry, rz], dtype=float))
            kalman_log.append((t, float(out["p_fault"]), int(out["p_fault"] >= 0.5)))

        step_counter[0] += 1
        if step_counter[0] % STEP != 0 or len(rmsw.vib_total) < WINDOW_SIZE:
            return
        feats = extract_features(
            np.fromiter(rmsw.vib_total, dtype=float, count=WINDOW_SIZE),
            np.fromiter(rmsw.vib_x, dtype=float, count=WINDOW_SIZE),
            np.fromiter(rmsw.vib_y, dtype=float, count=WINDOW_SIZE),
            np.fromiter(rmsw.vib_z, dtype=float, count=WINDOW_SIZE),
        )
        f_bin, t_bin, f_pos, t_pos = _process_window(
            feats, forest_bin_bundle, tree_bin_bundle, forest_pos_bundle, tree_pos_bundle
        )
        t = time.time() - start_time
        forest_bin_log.append((t, f_bin[0], f_bin[1], f_bin[2]))
        tree_bin_log.append((t, t_bin[0], t_bin[1], t_bin[2]))
        forest_pos_log.append((t, f_pos[0], f_pos[1]))
        tree_pos_log.append((t, t_pos[0], t_pos[1]))
        _append_smoothed(forest_bin_buf, forest_bin_smooth_log, t, f_bin)
        _append_smoothed(tree_bin_buf, tree_bin_smooth_log, t, t_bin)
        _append_smoothed(forest_pos_buf, forest_pos_smooth_log, t, f_pos)
        _append_smoothed(tree_pos_buf, tree_pos_smooth_log, t, t_pos)

    packet = XbusPacket(on_data_available=on_packet)
    try:
        while True:
            b = serial.read_byte()
            if b:
                packet.feed_byte(b)
    except KeyboardInterrupt:
        print("\nОстановлено.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    position_labels = list(forest_pos_bundle["class_labels"])
    out_dir = _save_artifacts(
        base,
        f"nn_v2_live_{ts}",
        vib_rows,
        kalman_log,
        forest_bin_log,
        forest_bin_smooth_log,
        tree_bin_log,
        tree_bin_smooth_log,
        forest_pos_log,
        forest_pos_smooth_log,
        tree_pos_log,
        tree_pos_smooth_log,
        position_labels,
    )
    print(f"Артефакты: {out_dir}")
    _maybe_export_and_metrics(
        base,
        out_dir,
        kalman_log,
        forest_bin_log,
        forest_bin_smooth_log,
        tree_bin_log,
        tree_bin_smooth_log,
        forest_pos_log,
        forest_pos_smooth_log,
        tree_pos_log,
        tree_pos_smooth_log,
    )
    return out_dir


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="NN v2.0 forest + tree (binary & position) + Kalman; "
        "labels are exported to datasets/set_03"
    )
    parser.add_argument("--base", type=Path, default=here)
    parser.add_argument("--calib", type=Path, default=here / "datasets" / "set_03")
    parser.add_argument("--offline", type=Path, default=None)
    parser.add_argument("--nn-forest-binary-model", type=Path, default=None)
    parser.add_argument("--nn-tree-binary-model", type=Path, default=None)
    parser.add_argument("--nn-forest-position-model", type=Path, default=None)
    parser.add_argument("--nn-tree-position-model", type=Path, default=None)
    args = parser.parse_args()

    base = args.base.resolve()
    forest_bin_path = _resolve_model(base, args.nn_forest_binary_model, DEFAULT_FOREST_BINARY_CANDIDATES)
    tree_bin_path = _resolve_model(base, args.nn_tree_binary_model, DEFAULT_TREE_BINARY_CANDIDATES)
    forest_pos_path = _resolve_model(base, args.nn_forest_position_model, DEFAULT_FOREST_POSITION_CANDIDATES)
    tree_pos_path = _resolve_model(base, args.nn_tree_position_model, DEFAULT_TREE_POSITION_CANDIDATES)
    print(f"NN forest binary:   {forest_bin_path}")
    print(f"NN tree   binary:   {tree_bin_path}")
    print(f"NN forest position: {forest_pos_path}")
    print(f"NN tree   position: {tree_pos_path}")
    print(f"Калибровка Калмана: {args.calib.resolve() / 'Normal_mod'}")
    print(f"Лог полёта будет экспортирован в datasets/{EXPORT_DATASET}")

    forest_bin_bundle = load_model(forest_bin_path)
    tree_bin_bundle = load_model(tree_bin_path)
    forest_pos_bundle = load_model(forest_pos_path)
    tree_pos_bundle = load_model(tree_pos_path)

    if args.offline is not None:
        _run_common_offline(
            base,
            args.calib.resolve(),
            args.offline.resolve(),
            forest_bin_bundle,
            tree_bin_bundle,
            forest_pos_bundle,
            tree_pos_bundle,
        )
    else:
        run_live(
            base,
            args.calib.resolve(),
            forest_bin_bundle,
            tree_bin_bundle,
            forest_pos_bundle,
            tree_pos_bundle,
        )


if __name__ == "__main__":
    main()
