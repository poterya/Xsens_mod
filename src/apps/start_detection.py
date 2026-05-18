#!/usr/bin/env python3
"""Flight test detector: MLP binary model.

Console contract during flight: print only "ПОЛОМКА" when the MLP reports
fault. Everything else is saved into runs/flight_test/<flight>/.
"""
from __future__ import annotations

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


import argparse
import contextlib
import io
import os
import sys
import time
import traceback
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from apps.all_detectors import _RMSWindow
from common.feature_extraction import extract_features
from common.realtime_detector import STEP, WINDOW_SIZE

DEFAULT_MLP_CANDIDATES = (
    "models/mlp/propeller_fault_mlp_keras_set05.pt",
)
OUTPUT_ROOT_NAME = "runs/flight_test"
SMOOTH_WIN = 5


def _resolve_model(base: Path, explicit: Path | None, candidates: tuple[str, ...]) -> Path:
    if explicit:
        path = explicit if explicit.is_absolute() else base / explicit
        if path.is_file():
            return path.resolve()
        raise SystemExit(f"Model not found: {path}")
    for name in candidates:
        path = base / name
        if path.is_file():
            return path.resolve()
    raise SystemExit("MLP model not found")


class MLPDetector(nn.Module):
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


def _feature_names() -> list[str]:
    zeros = np.zeros(WINDOW_SIZE, dtype=float)
    return list(extract_features(zeros, zeros, zeros, zeros).keys())


def _load_mlp_model(model_path: Path) -> dict:
    model = MLPDetector(in_features=80)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return {
        "classifier": model,
        "feature_names": _feature_names(),
        "class_labels": ["normal", "fault"],
    }


def _predict_binary(bundle: dict, feats: dict) -> tuple[str, float, float]:
    feature_names = list(bundle["feature_names"])
    x = torch.tensor(
        [[float(feats.get(name, 0.0)) for name in feature_names]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        p_fault = float(torch.sigmoid(bundle["classifier"](x))[0].item())
    if p_fault >= 0.5:
        return "fault", p_fault, p_fault
    return "normal", 1.0 - p_fault, p_fault


def _smooth_label(buf: deque[tuple[str, float]]) -> tuple[str, float] | None:
    if not buf:
        return None
    counts: dict[str, list[float]] = {}
    for label, conf in buf:
        counts.setdefault(label, []).append(float(conf))
    best_label = max(counts.items(), key=lambda kv: (len(kv[1]), np.mean(kv[1])))[0]
    return best_label, float(np.mean(counts[best_label]))


def _packet_hex(raw_packet) -> str:
    return "".join(b.hex().upper() for b in raw_packet)


def _csv_value(value, available: bool) -> str:
    return str(value) if available else ""


def _csv_vec(vec, available: bool) -> tuple[str, str, str]:
    if not available:
        return "", "", ""
    return f"{vec[0]:.9g}", f"{vec[1]:.9g}", f"{vec[2]:.9g}"


def _write_csv(path: Path, header: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def _save_vibration_plot(out_dir: Path, vib_rows) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    if vib_rows:
        data = np.array(vib_rows, dtype=float)
        t = data[:, 0]
        ax1.plot(t, data[:, 1], color="black", linewidth=1.4, label="total_vibration")
        for idx, label, color in (
            (2, "rms_x", "tab:blue"),
            (3, "rms_y", "tab:orange"),
            (4, "rms_z", "tab:green"),
        ):
            ax2.plot(t, data[:, idx], linewidth=1.1, label=label, color=color)
    else:
        ax1.text(0.5, 0.5, "no vibration samples", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_ylabel("total")
    ax1.grid(True, alpha=0.3)
    if ax1.lines:
        ax1.legend(loc="upper left")
    ax1.set_title("Vibration log")
    ax2.set_xlabel("time, s")
    ax2.set_ylabel("RMS")
    ax2.grid(True, alpha=0.3)
    if ax2.lines:
        ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "vibration_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_decision_plot(out_dir: Path, decision_rows, smooth_rows) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    if decision_rows:
        t = np.array([r[0] for r in decision_rows], dtype=float)
        p_fault = np.array([r[3] for r in decision_rows], dtype=float)
        label_idx = np.array([1 if r[1] == "fault" else 0 for r in decision_rows], dtype=float)
        ax1.plot(t, p_fault, color="tab:purple", linewidth=1.2, label="MLP p_fault")
        ax2.step(t, label_idx, where="post", color="tab:purple", linewidth=1.0, label="raw")
    else:
        ax1.text(0.5, 0.5, "no decisions", ha="center", va="center", transform=ax1.transAxes)
    if smooth_rows:
        st = np.array([r[0] for r in smooth_rows], dtype=float)
        slabel_idx = np.array([1 if r[1] == "fault" else 0 for r in smooth_rows], dtype=float)
        ax2.step(st, slabel_idx, where="post", color="tab:red", linewidth=1.2, label="smoothed")
    ax1.axhline(0.5, color="0.4", linestyle="--", linewidth=1, label="threshold 0.5")
    ax1.set_ylabel("p(FAULT)")
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, alpha=0.3)
    if ax1.lines:
        ax1.legend(loc="upper right")
    ax1.set_title("MLP binary decisions")

    ax2.set_xlabel("time, s")
    ax2.set_ylabel("0=normal\n1=fault")
    ax2.set_ylim(-0.1, 1.15)
    ax2.grid(True, alpha=0.3)
    if ax2.lines:
        ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "decision_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_artifacts(
    out_dir: Path,
    xsens_raw_rows,
    xsens_parsed_rows,
    vibration_rows,
    decision_rows,
    smooth_rows,
    console_rows,
    internal_rows,
) -> None:
    _write_csv(
        out_dir / "xsens_raw_packets.csv",
        "time_seconds,packet_hex",
        ((f"{t:.6f}", hx) for t, hx in xsens_raw_rows),
    )
    _write_csv(
        out_dir / "xsens_parsed.csv",
        (
            "time_seconds,packet_counter,sample_time_fine,"
            "acc_x,acc_y,acc_z,free_acc_x,free_acc_y,free_acc_z,"
            "gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,roll,pitch,yaw,"
            "temperature,status_word"
        ),
        xsens_parsed_rows,
    )
    _write_csv(
        out_dir / "vibration_log.csv",
        "time_seconds,total_vibration,rms_x,rms_y,rms_z",
        ((f"{t:.3f}", f"{total:.6f}", f"{rx:.6f}", f"{ry:.6f}", f"{rz:.6f}") for t, total, rx, ry, rz in vibration_rows),
    )
    _write_csv(
        out_dir / "decision_log.csv",
        "time_seconds,label,confidence,p_fault,console_output",
        ((f"{t:.4f}", label, f"{conf:.6f}", f"{pf:.6f}", out) for t, label, conf, pf, out in decision_rows),
    )
    _write_csv(
        out_dir / "decision_smoothed.csv",
        "time_seconds,label,confidence",
        ((f"{t:.4f}", label, f"{conf:.6f}") for t, label, conf in smooth_rows),
    )
    with open(out_dir / "console_output_log.txt", "w", encoding="utf-8") as f:
        for t, text in console_rows:
            f.write(f"{t:.4f},{text}\n")
    with open(out_dir / "internal_log.txt", "w", encoding="utf-8") as f:
        for t, text in internal_rows:
            f.write(f"{t:.4f},{text}\n")
    (out_dir / "README.txt").write_text(
        "\n".join(
            [
                "Detector: MLP binary model.",
                "Console prints only the word 'ПОЛОМКА' when the raw MLP prediction is fault.",
                "xsens_raw_packets.csv: raw Xbus packet hex.",
                "xsens_parsed.csv: parsed Xsens fields available in each packet.",
                "vibration_log.csv + vibration_plot.png: RMS vibration stream.",
                "decision_log.csv + decision_plot.png: neural-network decisions.",
                "console_output_log.txt: exact fault messages printed to console.",
                "internal_log.txt: service/errors captured without printing to console.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _save_vibration_plot(out_dir, vibration_rows)
    _save_decision_plot(out_dir, decision_rows, smooth_rows)


class QuietXbusPacket:
    """XbusPacket-compatible parser without checksum prints."""

    def __init__(self, on_data_available=None):
        self.on_data_available = on_data_available
        self.reset()

    def reset(self):
        self.buffer = []
        self.expected_length = 0
        self.length_valid = False

    def feed_byte(self, byte):
        if len(self.buffer) == 0 and byte == b"\xfa":
            self.buffer.append(byte)
            return
        if len(self.buffer) == 1 and byte == b"\xff":
            self.buffer.append(byte)
            return
        if len(self.buffer) == 2 and byte == b"\x36":
            self.buffer.append(byte)
            return
        if len(self.buffer) == 3:
            self.buffer.append(byte)
            self.expected_length = byte[0]
            self.length_valid = True
            return
        if self.length_valid and len(self.buffer) >= 4:
            self.buffer.append(byte)
            total_length = 3 + 1 + self.expected_length + 1
            if len(self.buffer) == total_length:
                if self.validate_checksum() and self.on_data_available is not None:
                    self.on_data_available(self.buffer)
                self.reset()

    def validate_checksum(self):
        if not self.length_valid or len(self.buffer) != (3 + 1 + self.expected_length + 1):
            return False
        total = sum(byte[0] for byte in self.buffer[1:-1])
        checksum = (-total) & 0xFF
        return checksum == self.buffer[-1][0]


def _append_parsed_row(rows, t: float, xbus) -> None:
    acc = _csv_vec(xbus.acc, xbus.accAvailable)
    free_acc = _csv_vec(xbus.freeAcc, xbus.freeAccAvailable)
    rot = _csv_vec(xbus.rot, xbus.rotAvailable)
    mag = _csv_vec(xbus.mag, xbus.magAvailable)
    euler = _csv_vec(xbus.euler, xbus.eulerAvailable)
    rows.append(
        (
            f"{t:.6f}",
            _csv_value(xbus.packetCounter, xbus.packetCounterAvailable),
            _csv_value(xbus.sampleTimeFine, xbus.sampleTimeFineAvailable),
            *acc,
            *free_acc,
            *rot,
            *mag,
            *euler,
            _csv_value(xbus.temperature, xbus.temperatureAvailable),
            _csv_value(xbus.statusWord, xbus.statusWordAvailable),
        )
    )


def run_live(base: Path, model_bundle: dict, port: str, baudrate: int) -> Path:
    from xsens.DataPacketParser import DataPacketParser, XsDataPacket
    from xsens.SerialHandler import SerialHandler

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / OUTPUT_ROOT_NAME / f"start_detection_live_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rmsw = _RMSWindow(WINDOW_SIZE)
    xsens_raw_rows = []
    xsens_parsed_rows = []
    vibration_rows = []
    decision_rows = []
    smooth_rows = []
    console_rows = []
    internal_rows = []
    smooth_buf: deque[tuple[str, float]] = deque(maxlen=SMOOTH_WIN)
    step_counter = [0]
    start_time = time.time()

    def now() -> float:
        return time.time() - start_time

    def log_internal(text: str) -> None:
        internal_rows.append((now(), text.replace("\n", "\\n")))

    def on_packet(raw_packet):
        t = now()
        xsens_raw_rows.append((t, _packet_hex(raw_packet)))
        xbus = XsDataPacket()
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            try:
                DataPacketParser.parse_data_packet(raw_packet, xbus)
            except Exception:
                log_internal("parse_error: " + traceback.format_exc())
                return
        captured_text = captured.getvalue().strip()
        if captured_text:
            log_internal("parser_stdout: " + captured_text)

        _append_parsed_row(xsens_parsed_rows, t, xbus)
        if not xbus.accAvailable:
            return

        n_before = len(rmsw.vib_total)
        rmsw.add(xbus.acc[0], xbus.acc[1], xbus.acc[2])
        if len(rmsw.vib_total) > n_before:
            total = float(rmsw.vib_total[-1])
            rx = float(rmsw.vib_x[-1])
            ry = float(rmsw.vib_y[-1])
            rz = float(rmsw.vib_z[-1])
            vibration_rows.append((t, total, rx, ry, rz))

        step_counter[0] += 1
        if step_counter[0] % STEP != 0 or len(rmsw.vib_total) < WINDOW_SIZE:
            return

        feats = extract_features(
            np.fromiter(rmsw.vib_total, dtype=float, count=WINDOW_SIZE),
            np.fromiter(rmsw.vib_x, dtype=float, count=WINDOW_SIZE),
            np.fromiter(rmsw.vib_y, dtype=float, count=WINDOW_SIZE),
            np.fromiter(rmsw.vib_z, dtype=float, count=WINDOW_SIZE),
        )
        label, conf, p_fault = _predict_binary(model_bundle, feats)
        console_output = "ПОЛОМКА" if label == "fault" else ""
        decision_rows.append((t, label, conf, p_fault, console_output))
        smooth_buf.append((label, conf))
        sm = _smooth_label(smooth_buf)
        if sm is not None:
            smooth_rows.append((t, sm[0], sm[1]))
        if console_output:
            print(console_output, flush=True)
            console_rows.append((t, console_output))

    try:
        serial = SerialHandler(port, baudrate)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            serial.send_with_checksum(bytes.fromhex("FA FF 30 00"))
        if captured.getvalue().strip():
            log_internal("serial_stdout: " + captured.getvalue().strip())
        time.sleep(0.1)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            serial.send_with_checksum(bytes.fromhex(
                "FA FF C0 20 10 20 FF FF 10 60 FF FF "
                "20 30 00 64 40 20 00 64 40 30 00 64 "
                "80 20 00 64 C0 20 00 64 E0 20 FF FF"
            ))
        if captured.getvalue().strip():
            log_internal("serial_stdout: " + captured.getvalue().strip())
        time.sleep(0.1)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            serial.send_with_checksum(bytes.fromhex("FA FF 10 00"))
        if captured.getvalue().strip():
            log_internal("serial_stdout: " + captured.getvalue().strip())

        packet = QuietXbusPacket(on_data_available=on_packet)
        while True:
            try:
                with contextlib.redirect_stdout(io.StringIO()) as captured:
                    b = serial.read_byte()
                if captured.getvalue().strip():
                    log_internal("serial_stdout: " + captured.getvalue().strip())
            except Exception:
                log_internal("serial_read_error: " + traceback.format_exc())
                continue
            if b:
                packet.feed_byte(b)
    except KeyboardInterrupt:
        pass
    except Exception:
        log_internal("fatal_error: " + traceback.format_exc())
    finally:
        _save_artifacts(
            out_dir,
            xsens_raw_rows,
            xsens_parsed_rows,
            vibration_rows,
            decision_rows,
            smooth_rows,
            console_rows,
            internal_rows,
        )
    return out_dir


def run_offline(base: Path, model_bundle: dict, csv_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    cols = ["total_vibration", "rms_x", "rms_y", "rms_z"]
    if not set(cols).issubset(df.columns):
        raise SystemExit(f"CSV must contain columns: {cols}")
    times = df["time_seconds"].to_numpy(dtype=float) if "time_seconds" in df.columns else np.arange(len(df)) * 0.01

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / OUTPUT_ROOT_NAME / f"start_detection_offline_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    vib_total: deque = deque(maxlen=WINDOW_SIZE)
    vib_x: deque = deque(maxlen=WINDOW_SIZE)
    vib_y: deque = deque(maxlen=WINDOW_SIZE)
    vib_z: deque = deque(maxlen=WINDOW_SIZE)
    vibration_rows = []
    decision_rows = []
    smooth_rows = []
    console_rows = []
    smooth_buf: deque[tuple[str, float]] = deque(maxlen=SMOOTH_WIN)
    step_i = 0

    for row_idx, (_, row) in enumerate(df.iterrows()):
        z = row[cols].to_numpy(dtype=float)
        t = float(times[row_idx]) if row_idx < len(times) else row_idx * 0.01
        vib_total.append(z[0]); vib_x.append(z[1]); vib_y.append(z[2]); vib_z.append(z[3])
        vibration_rows.append((t, float(z[0]), float(z[1]), float(z[2]), float(z[3])))
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
        label, conf, p_fault = _predict_binary(model_bundle, feats)
        console_output = "ПОЛОМКА" if label == "fault" else ""
        decision_rows.append((t, label, conf, p_fault, console_output))
        smooth_buf.append((label, conf))
        sm = _smooth_label(smooth_buf)
        if sm is not None:
            smooth_rows.append((t, sm[0], sm[1]))
        if console_output:
            print(console_output, flush=True)
            console_rows.append((t, console_output))

    _save_artifacts(
        out_dir,
        [],
        [],
        vibration_rows,
        decision_rows,
        smooth_rows,
        console_rows,
        [(0.0, f"offline_source={csv_path}")],
    )
    return out_dir


def main() -> None:
    here = _PROJECT_ROOT
    parser = argparse.ArgumentParser(
        description="Flight test: MLP binary detector. Console prints only ПОЛОМКА."
    )
    parser.add_argument("--base", type=Path, default=here)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--offline", type=Path, default=None)
    args = parser.parse_args()

    base = args.base.resolve()
    model_path = _resolve_model(base, args.model, DEFAULT_MLP_CANDIDATES)
    model_bundle = _load_mlp_model(model_path)

    if args.offline:
        run_offline(base, model_bundle, args.offline.resolve())
    else:
        run_live(base, model_bundle, args.port, args.baudrate)


if __name__ == "__main__":
    main()
