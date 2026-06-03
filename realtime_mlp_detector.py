"""Реал-тайм детектор поломки винта по Xsens с использованием MLP.

Источник данных — IMU Xsens по UART (SerialHandler + XbusPacket +
DataPacketParser). Из сырых ускорений формируется окно RMS-сигналов,
далее вычисляются 46 признаков из общего модуля features.py и
подаются в обученный Pipeline (StandardScaler + MLPClassifier),
сохранённый в models/method_mlp.pkl. Выход — флаг НОРМА / ПОЛОМКА
с EMA-сглаживанием и гистерезисом. По Ctrl+C сохраняются лог решений,
лог вибрации и графики.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib
if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from features import (
    WINDOW_SIZE,
    STEP,
    extract_features,
)
from SerialHandler import SerialHandler
from DataPacketParser import DataPacketParser, XsDataPacket
from XbusPacket import XbusPacket

RMS_WINDOW = 50
LATENCY_BUDGET_MS = 100.0
EMA_ALPHA = 0.30
FAULT_ON = 0.60
FAULT_OFF = 0.40
DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 115200
MODEL_PATH = HERE / "models" / "method_mlp.pkl"
OUT_ROOT = HERE / "runs" / "realtime_mlp"

MTI30_GO_CONFIG = bytes.fromhex("FA FF 30 00")
MTI30_GO_MEAS = bytes.fromhex("FA FF 10 00")
MTI30_CONFIG = bytes.fromhex(
    "FA FF C0 20 10 20 FF FF 10 60 FF FF "
    "20 30 00 64 40 20 00 64 40 30 00 64 "
    "80 20 00 64 C0 20 00 64 E0 20 FF FF"
)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.prop_cycle": plt.cycler(color=["black", "0.35", "0.6"]),
})


def _load_pipeline(path: Path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], list(bundle["feature_names"])


def _rms_step(buf_x, buf_y, buf_z):
    x = np.fromiter(buf_x, dtype=float, count=len(buf_x))
    y = np.fromiter(buf_y, dtype=float, count=len(buf_y))
    z = np.fromiter(buf_z, dtype=float, count=len(buf_z))
    mx, my, mz = x.mean(), y.mean(), z.mean()
    vx, vy, vz = x - mx, y - my, z - mz
    rx = float(np.sqrt(np.mean(vx * vx)))
    ry = float(np.sqrt(np.mean(vy * vy)))
    rz = float(np.sqrt(np.mean(vz * vz)))
    total = float(np.sqrt(rx * rx + ry * ry + rz * rz))
    return total, rx, ry, rz


def _make_feature_vector(buf_total, buf_x, buf_y, buf_z, feature_names):
    total = np.fromiter(buf_total, dtype=float, count=len(buf_total))
    rx = np.fromiter(buf_x, dtype=float, count=len(buf_x))
    ry = np.fromiter(buf_y, dtype=float, count=len(buf_y))
    rz = np.fromiter(buf_z, dtype=float, count=len(buf_z))
    feats = extract_features(total, rx, ry, rz)
    return np.array([[feats.get(n, 0.0) for n in feature_names]], dtype=float)


def _configure_xsens(serial: SerialHandler) -> None:
    serial.send_with_checksum(MTI30_GO_CONFIG)
    print("Xsens: переход в config mode...")
    time.sleep(0.1)
    serial.send_with_checksum(MTI30_CONFIG)
    print("Xsens: выходные данные настроены (MTi-30, 100 Гц)...")
    time.sleep(0.1)
    serial.send_with_checksum(MTI30_GO_MEAS)
    print("Xsens: переход в measurement mode.")


def _save_session(out_dir: Path, decisions, raw_rms) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if decisions:
        arr = np.array(decisions, dtype=float)
        np.savetxt(
            out_dir / "decision_log.csv",
            arr,
            delimiter=",",
            header="time_s,proba,proba_ema,flag,latency_ms",
            comments="",
            fmt=["%.4f", "%.6f", "%.6f", "%d", "%.2f"],
        )
        fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), dpi=140, sharex=True)
        ax[0].plot(arr[:, 0], arr[:, 1], color="0.55", linewidth=0.9, label="вероятность (поломка)")
        ax[0].plot(arr[:, 0], arr[:, 2], color="black", linewidth=1.4, label="EMA")
        ax[0].axhline(FAULT_ON, color="0.4", linestyle="--", linewidth=0.8)
        ax[0].axhline(FAULT_OFF, color="0.4", linestyle=":", linewidth=0.8)
        ax[0].set_ylabel("Вероятность поломки")
        ax[0].set_ylim(-0.02, 1.02)
        ax[0].legend(loc="upper right")
        ax[0].set_title("Реал-тайм детектирование MLP")
        ax[1].step(arr[:, 0], arr[:, 3], where="post", color="black", linewidth=1.3)
        ax[1].set_yticks([0, 1])
        ax[1].set_yticklabels(["норма", "поломка"])
        ax[1].set_xlabel("Время, с")
        fig.tight_layout()
        fig.savefig(out_dir / "decision_plot.png", bbox_inches="tight")
        plt.close(fig)
    if raw_rms:
        rms = np.array(raw_rms, dtype=float)
        np.savetxt(
            out_dir / "vibration_log.csv",
            rms,
            delimiter=",",
            header="time_seconds,total_vibration,rms_x,rms_y,rms_z",
            comments="",
            fmt="%.6f",
        )
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
        ax.plot(rms[:, 0], rms[:, 1], color="black", linewidth=1.2, label="total_vibration")
        ax.plot(rms[:, 0], rms[:, 2], color="0.45", linewidth=0.9, linestyle="--", label="rms_x")
        ax.plot(rms[:, 0], rms[:, 3], color="0.45", linewidth=0.9, linestyle="-.", label="rms_y")
        ax.plot(rms[:, 0], rms[:, 4], color="0.45", linewidth=0.9, linestyle=":", label="rms_z")
        ax.set_xlabel("Время, с")
        ax.set_ylabel("Амплитуда")
        ax.set_title("Вибрация в сессии")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "vibration_plot.png", bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Реал-тайм MLP-детектор поломки винта по Xsens")
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Модель не найдена: {model_path}", file=sys.stderr)
        return 1
    pipe, feature_names = _load_pipeline(model_path)
    print(f"Модель: {model_path}")
    print(f"Признаков: {len(feature_names)}  Окно: {WINDOW_SIZE} отсчётов  Шаг: {STEP}")
    print(f"Бюджет на цикл: {LATENCY_BUDGET_MS:.0f} мс")
    print(f"Подключение к Xsens: {args.port} @ {args.baud}")

    serial = SerialHandler(args.port, args.baud)
    _configure_xsens(serial)
    print("=" * 80)

    acc_x: deque = deque(maxlen=RMS_WINDOW)
    acc_y: deque = deque(maxlen=RMS_WINDOW)
    acc_z: deque = deque(maxlen=RMS_WINDOW)
    rms_total: deque = deque(maxlen=WINDOW_SIZE)
    rms_x: deque = deque(maxlen=WINDOW_SIZE)
    rms_y: deque = deque(maxlen=WINDOW_SIZE)
    rms_z: deque = deque(maxlen=WINDOW_SIZE)

    state = {
        "samples_since_predict": 0,
        "proba_ema": 0.0,
        "fault_flag": 0,
        "latency_violations": 0,
    }
    decisions: list[tuple[float, float, float, int, float]] = []
    raw_rms: list[tuple[float, float, float, float, float]] = []
    t0 = time.monotonic()

    def on_packet(raw_packet) -> None:
        t_pkt = time.perf_counter()
        xs_data = XsDataPacket()
        DataPacketParser.parse_data_packet(raw_packet, xs_data)
        if not xs_data.accAvailable:
            return
        acc_x.append(float(xs_data.acc[0]))
        acc_y.append(float(xs_data.acc[1]))
        acc_z.append(float(xs_data.acc[2]))
        if len(acc_x) < RMS_WINDOW:
            return
        total, rx, ry, rz = _rms_step(acc_x, acc_y, acc_z)
        t_now = time.monotonic() - t0
        rms_total.append(total)
        rms_x.append(rx)
        rms_y.append(ry)
        rms_z.append(rz)
        raw_rms.append((t_now, total, rx, ry, rz))
        state["samples_since_predict"] += 1
        if len(rms_total) < WINDOW_SIZE or state["samples_since_predict"] < STEP:
            return
        state["samples_since_predict"] = 0
        t_predict = time.perf_counter()
        x_vec = _make_feature_vector(rms_total, rms_x, rms_y, rms_z, feature_names)
        proba = float(pipe.predict_proba(x_vec)[0, 1])
        state["proba_ema"] = EMA_ALPHA * proba + (1.0 - EMA_ALPHA) * state["proba_ema"]
        if state["fault_flag"] == 0 and state["proba_ema"] >= FAULT_ON:
            state["fault_flag"] = 1
        elif state["fault_flag"] == 1 and state["proba_ema"] <= FAULT_OFF:
            state["fault_flag"] = 0
        latency_ms = (time.perf_counter() - t_predict) * 1000.0
        if latency_ms > LATENCY_BUDGET_MS:
            state["latency_violations"] += 1
        decisions.append((t_now, proba, state["proba_ema"], state["fault_flag"], latency_ms))
        tag = "\033[91mПОЛОМКА\033[0m" if state["fault_flag"] == 1 else "\033[92mНОРМА  \033[0m"
        warn = "" if latency_ms <= LATENCY_BUDGET_MS else "  ! over budget"
        sys.stdout.write(
            f"\rt={t_now:7.2f}s  vib={total:.4f}  "
            f"p={proba:.3f}  ema={state['proba_ema']:.3f}  "
            f"{tag}  lat={latency_ms:5.1f} ms{warn}        "
        )
        sys.stdout.flush()

    packet = XbusPacket(on_data_available=on_packet)

    print("\nЗапуск. Ctrl+C — остановить и сохранить отчёт.\n")
    try:
        while True:
            byte = serial.read_byte()
            if byte:
                packet.feed_byte(byte)
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
    finally:
        print()
        if state["latency_violations"]:
            print(f"Превышений бюджета {LATENCY_BUDGET_MS:.0f} мс: {state['latency_violations']}")

    out_dir = Path(args.out) if args.out else (OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S"))
    _save_session(out_dir, decisions, raw_rms)
    print(f"Отчёт сохранён в {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
