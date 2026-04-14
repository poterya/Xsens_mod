#!/usr/bin/env python3
"""
Детекция поломки винта квадрокоптера в реальном времени через Xsens.

Загружает обученную модель (propeller_fault_model.pkl), подключается к Xsens
по serial, копит скользящее окно вибрации и каждые WINDOW_SIZE отсчётов
выдаёт вердикт: NORMAL / DEFORMED (поломка винта).
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from SerialHandler import SerialHandler
from XbusPacket import XbusPacket
from DataPacketParser import DataPacketParser, XsDataPacket


WINDOW_SIZE = 50
STEP = 10
CONFIDENCE_QUEUE_SIZE = 5


def load_model(model_path: Path) -> dict:
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    required = {"classifier", "feature_names", "window_size"}
    if not required.issubset(bundle.keys()):
        raise ValueError(f"Модель не содержит обязательных ключей: {required - bundle.keys()}")
    return bundle


def extract_features(total: np.ndarray, rms_x: np.ndarray,
                     rms_y: np.ndarray, rms_z: np.ndarray) -> dict:
    feats: dict = {}
    for name, v in [("total_vibration", total), ("rms_x", rms_x),
                    ("rms_y", rms_y), ("rms_z", rms_z)]:
        feats[f"{name}_mean"] = np.mean(v)
        feats[f"{name}_std"] = np.std(v)
        feats[f"{name}_max"] = np.max(v)
        feats[f"{name}_min"] = np.min(v)
        feats[f"{name}_range"] = np.ptp(v)
        feats[f"{name}_median"] = np.median(v)

    feats["ratio_x_total"] = feats["rms_x_mean"] / max(feats["total_vibration_mean"], 1e-9)
    feats["ratio_y_total"] = feats["rms_y_mean"] / max(feats["total_vibration_mean"], 1e-9)
    feats["ratio_z_total"] = feats["rms_z_mean"] / max(feats["total_vibration_mean"], 1e-9)

    if len(total) >= 2:
        feats["total_vibration_diff_mean"] = np.mean(np.abs(np.diff(total)))
        feats["total_vibration_diff_max"] = np.max(np.abs(np.diff(total)))
    else:
        feats["total_vibration_diff_mean"] = 0.0
        feats["total_vibration_diff_max"] = 0.0

    return feats


class RealtimeFaultDetector:
    def __init__(self, model_bundle: dict, window_size: int = WINDOW_SIZE):
        self.clf = model_bundle["classifier"]
        self.feature_names = model_bundle["feature_names"]
        self.window_size = window_size

        self.acc_x: deque = deque(maxlen=window_size)
        self.acc_y: deque = deque(maxlen=window_size)
        self.acc_z: deque = deque(maxlen=window_size)

        self.vib_total: deque = deque(maxlen=window_size)
        self.vib_x: deque = deque(maxlen=window_size)
        self.vib_y: deque = deque(maxlen=window_size)
        self.vib_z: deque = deque(maxlen=window_size)

        self.sample_count = 0
        self.prediction_log: list[tuple[float, int, float]] = []
        self.confidence_queue: deque = deque(maxlen=CONFIDENCE_QUEUE_SIZE)
        self.start_time = time.time()

    def add_accel(self, ax: float, ay: float, az: float) -> None:
        self.acc_x.append(ax)
        self.acc_y.append(ay)
        self.acc_z.append(az)
        self.sample_count += 1

        if len(self.acc_x) >= self.window_size:
            mean_x = np.mean(self.acc_x)
            mean_y = np.mean(self.acc_y)
            mean_z = np.mean(self.acc_z)

            vx = np.array(self.acc_x) - mean_x
            vy = np.array(self.acc_y) - mean_y
            vz = np.array(self.acc_z) - mean_z

            rms_x = np.sqrt(np.mean(vx ** 2))
            rms_y = np.sqrt(np.mean(vy ** 2))
            rms_z = np.sqrt(np.mean(vz ** 2))
            total = np.sqrt(rms_x ** 2 + rms_y ** 2 + rms_z ** 2)

            self.vib_total.append(total)
            self.vib_x.append(rms_x)
            self.vib_y.append(rms_y)
            self.vib_z.append(rms_z)

    def predict(self) -> tuple[int, float] | None:
        if len(self.vib_total) < self.window_size:
            return None

        feats = extract_features(
            np.array(self.vib_total),
            np.array(self.vib_x),
            np.array(self.vib_y),
            np.array(self.vib_z),
        )

        x_vec = np.array([[feats.get(f, 0.0) for f in self.feature_names]])
        pred = self.clf.predict(x_vec)[0]
        proba = self.clf.predict_proba(x_vec)[0]
        confidence = float(proba[pred])

        self.confidence_queue.append(pred)
        elapsed = time.time() - self.start_time
        self.prediction_log.append((elapsed, pred, confidence))
        return int(pred), confidence

    @property
    def smoothed_label(self) -> int | None:
        if len(self.confidence_queue) < CONFIDENCE_QUEUE_SIZE:
            return None
        return int(np.round(np.mean(self.confidence_queue)))


def display_status(pred: int, confidence: float, smoothed: int | None,
                   rms_total: float | None) -> None:
    label = "DEFORMED" if pred == 1 else "NORMAL"
    color = "\033[91m" if pred == 1 else "\033[92m"
    reset = "\033[0m"
    smooth_str = ""
    if smoothed is not None:
        s_label = "DEFORMED" if smoothed == 1 else "NORMAL"
        s_color = "\033[91m" if smoothed == 1 else "\033[92m"
        smooth_str = f"  сглаж: {s_color}{s_label}{reset}"
    vib_str = f"  vib={rms_total:.4f}" if rms_total else ""
    print(f"{color}[{label}]{reset} уверенность={confidence:.2f}{smooth_str}{vib_str}")


def main() -> None:
    base = Path(__file__).resolve().parent
    model_path = base / "propeller_fault_model.pkl"
    if not model_path.exists():
        print(f"Модель не найдена: {model_path}", file=sys.stderr)
        print("Сначала запустите: python3 train_detector.py", file=sys.stderr)
        sys.exit(1)

    bundle = load_model(model_path)
    detector = RealtimeFaultDetector(bundle)
    print(f"Модель загружена: {model_path.name}")
    print(f"Окно: {WINDOW_SIZE}, шаг предсказания: {STEP}, сглаживание: {CONFIDENCE_QUEUE_SIZE}")
    print("=" * 60)

    serial = SerialHandler("/dev/ttyUSB0", 115200)

    go_to_config = bytes.fromhex("FA FF 30 00")
    go_to_measurement = bytes.fromhex("FA FF 10 00")

    serial.send_with_checksum(go_to_config)
    print("Config mode...")
    time.sleep(0.1)

    config_mti30 = bytes.fromhex(
        "FA FF C0 20 10 20 FF FF 10 60 FF FF "
        "20 30 00 64 40 20 00 64 40 30 00 64 "
        "80 20 00 64 C0 20 00 64 E0 20 FF FF"
    )
    serial.send_with_checksum(config_mti30)
    print("Output configured (MTi-30, 100 Hz)...")
    time.sleep(0.1)

    serial.send_with_checksum(go_to_measurement)
    print("Measurement mode. Ctrl+C to stop.\n")

    step_counter = [0]
    last_total = [None]

    def on_packet(raw_packet):
        xbus_data = XsDataPacket()
        DataPacketParser.parse_data_packet(raw_packet, xbus_data)

        if not xbus_data.accAvailable:
            return

        detector.add_accel(xbus_data.acc[0], xbus_data.acc[1], xbus_data.acc[2])

        if len(detector.vib_total) > 0:
            last_total[0] = float(detector.vib_total[-1])

        step_counter[0] += 1
        if step_counter[0] % STEP != 0:
            return

        result = detector.predict()
        if result is None:
            return

        pred, conf = result
        display_status(pred, conf, detector.smoothed_label, last_total[0])

    packet = XbusPacket(on_data_available=on_packet)

    try:
        while True:
            byte = serial.read_byte()
            if byte:
                packet.feed_byte(byte)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Остановлено.")
        if detector.prediction_log:
            preds = [p[1] for p in detector.prediction_log]
            n_normal = preds.count(0)
            n_deformed = preds.count(1)
            print(f"Предсказаний: {len(preds)} (Normal={n_normal}, Deformed={n_deformed})")
            ratio = n_deformed / max(len(preds), 1)
            verdict = "DEFORMED" if ratio > 0.5 else "NORMAL"
            print(f"Итоговый вердикт сессии: {verdict} (deformed ratio={ratio:.2f})")

            log_dir = base / "detection_logs"
            log_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"detection_{ts}.csv"
            with open(log_path, "w") as f:
                f.write("elapsed_s,prediction,confidence\n")
                for t, p, c in detector.prediction_log:
                    f.write(f"{t:.3f},{p},{c:.4f}\n")
            print(f"Лог сохранён: {log_path}")


if __name__ == "__main__":
    main()
