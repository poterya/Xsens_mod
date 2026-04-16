#!/usr/bin/env python3
"""
Детекция поломки винта в реальном времени (Random Forest + FFT-признаки).

Загружает propeller_fault_rf.pkl, подключается к Xsens по serial,
выдаёт вердикт NORMAL / DEFORMED.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from SerialHandler import SerialHandler
from XbusPacket import XbusPacket
from DataPacketParser import DataPacketParser, XsDataPacket


WINDOW_SIZE = 50
STEP = 10
CONFIDENCE_QUEUE_SIZE = 5
SAMPLE_RATE_HZ = 100
DEFAULT_DEFORMED_THRESHOLD = 0.60
DEFAULT_SESSION_THRESHOLD = 0.20
DEFAULT_EMA_ALPHA = 0.25
DEFAULT_PEAK_THRESHOLD = 0.58


def load_model(model_path: Path) -> dict:
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    required = {"classifier", "feature_names", "window_size"}
    if not required.issubset(bundle.keys()):
        raise ValueError(f"Модель не содержит: {required - bundle.keys()}")
    return bundle


def _fft_features(signal: np.ndarray, prefix: str) -> dict:
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal - np.mean(signal)))
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE_HZ)

    power = fft_vals ** 2
    total_power = np.sum(power) + 1e-12

    peak_idx = np.argmax(fft_vals[1:]) + 1
    dominant_freq = freqs[peak_idx]
    dominant_mag = fft_vals[peak_idx]

    p = power / total_power
    spectral_entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    spectral_centroid = np.sum(freqs * power) / total_power

    return {
        f"{prefix}_dom_freq": dominant_freq,
        f"{prefix}_dom_mag": dominant_mag,
        f"{prefix}_spectral_energy": total_power,
        f"{prefix}_spectral_entropy": spectral_entropy,
        f"{prefix}_spectral_centroid": spectral_centroid,
    }


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
        feats[f"{name}_q25"] = np.percentile(v, 25)
        feats[f"{name}_q75"] = np.percentile(v, 75)
        feats[f"{name}_iqr"] = feats[f"{name}_q75"] - feats[f"{name}_q25"]
        feats[f"{name}_skew"] = float(pd.Series(v).skew())
        feats[f"{name}_kurtosis"] = float(pd.Series(v).kurtosis())
        feats.update(_fft_features(v, name))

    t_mean = max(feats["total_vibration_mean"], 1e-9)
    feats["ratio_x_total"] = feats["rms_x_mean"] / t_mean
    feats["ratio_y_total"] = feats["rms_y_mean"] / t_mean
    feats["ratio_z_total"] = feats["rms_z_mean"] / t_mean

    if len(total) >= 2:
        d = np.abs(np.diff(total))
        feats["total_vibration_diff_mean"] = np.mean(d)
        feats["total_vibration_diff_max"] = np.max(d)
        feats["total_vibration_diff_std"] = np.std(d)
    else:
        feats["total_vibration_diff_mean"] = 0.0
        feats["total_vibration_diff_max"] = 0.0
        feats["total_vibration_diff_std"] = 0.0

    feats["xy_correlation"] = float(np.corrcoef(rms_x, rms_y)[0, 1]) if len(rms_x) > 1 else 0.0
    feats["xz_correlation"] = float(np.corrcoef(rms_x, rms_z)[0, 1]) if len(rms_x) > 1 else 0.0
    feats["yz_correlation"] = float(np.corrcoef(rms_y, rms_z)[0, 1]) if len(rms_y) > 1 else 0.0

    return feats


class RealtimeRFDetector:
    def __init__(
        self,
        model_bundle: dict,
        window_size: int = WINDOW_SIZE,
        deformed_threshold: float = DEFAULT_DEFORMED_THRESHOLD,
        session_threshold: float = DEFAULT_SESSION_THRESHOLD,
        peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        invert_labels: bool = False,
    ):
        self.clf = model_bundle["classifier"]
        self.feature_names = model_bundle["feature_names"]
        self.window_size = window_size
        self.deformed_threshold = deformed_threshold
        self.session_threshold = session_threshold
        self.peak_threshold = peak_threshold
        self.ema_alpha = ema_alpha
        self.invert_labels = invert_labels

        self.acc_x: deque = deque(maxlen=window_size)
        self.acc_y: deque = deque(maxlen=window_size)
        self.acc_z: deque = deque(maxlen=window_size)

        self.vib_total: deque = deque(maxlen=window_size)
        self.vib_x: deque = deque(maxlen=window_size)
        self.vib_y: deque = deque(maxlen=window_size)
        self.vib_z: deque = deque(maxlen=window_size)

        self.sample_count = 0
        self.prediction_log: list[tuple[float, int, float, float, float]] = []
        self.confidence_queue: deque = deque(maxlen=CONFIDENCE_QUEUE_SIZE)
        self.start_time = time.time()
        self.p_deformed_ema: float | None = None

    @property
    def deformed_class_index(self) -> int:
        classes = getattr(self.clf, "classes_", np.array([0, 1]))
        matches = np.where(classes == 1)[0]
        return int(matches[0]) if len(matches) else 1

    def add_accel(self, ax: float, ay: float, az: float) -> None:
        self.acc_x.append(ax)
        self.acc_y.append(ay)
        self.acc_z.append(az)
        self.sample_count += 1

        if len(self.acc_x) >= self.window_size:
            arr_x = np.array(self.acc_x)
            arr_y = np.array(self.acc_y)
            arr_z = np.array(self.acc_z)

            vx = arr_x - np.mean(arr_x)
            vy = arr_y - np.mean(arr_y)
            vz = arr_z - np.mean(arr_z)

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
        nan_mask = np.isnan(x_vec)
        if np.any(nan_mask):
            x_vec[nan_mask] = 0.0

        proba = self.clf.predict_proba(x_vec)[0]
        p_deformed_raw = float(proba[self.deformed_class_index])
        p_deformed = 1.0 - p_deformed_raw if self.invert_labels else p_deformed_raw

        if self.p_deformed_ema is None:
            self.p_deformed_ema = p_deformed
        else:
            self.p_deformed_ema = self.ema_alpha * p_deformed + (1.0 - self.ema_alpha) * self.p_deformed_ema

        pred = 1 if self.p_deformed_ema >= self.deformed_threshold else 0
        confidence = self.p_deformed_ema if pred == 1 else (1.0 - self.p_deformed_ema)

        self.confidence_queue.append(pred)
        elapsed = time.time() - self.start_time
        self.prediction_log.append((elapsed, pred, confidence, p_deformed, self.p_deformed_ema))
        return int(pred), confidence

    @property
    def smoothed_label(self) -> int | None:
        if len(self.confidence_queue) < CONFIDENCE_QUEUE_SIZE:
            return None
        return int(np.round(np.mean(self.confidence_queue)))


def display_status(pred: int, confidence: float, smoothed: int | None,
                   rms_total: float | None, p_deformed: float | None = None,
                   p_deformed_ema: float | None = None) -> None:
    label = "DEFORMED" if pred == 1 else "NORMAL"
    color = "\033[91m" if pred == 1 else "\033[92m"
    reset = "\033[0m"
    smooth_str = ""
    if smoothed is not None:
        s_label = "DEFORMED" if smoothed == 1 else "NORMAL"
        s_color = "\033[91m" if smoothed == 1 else "\033[92m"
        smooth_str = f"  сглаж: {s_color}{s_label}{reset}"
    vib_str = f"  vib={rms_total:.4f}" if rms_total else ""
    p_def_str = f"  p_def={p_deformed:.2f}" if p_deformed is not None else ""
    p_ema_str = f"  p_ema={p_deformed_ema:.2f}" if p_deformed_ema is not None else ""
    print(f"{color}[{label}]{reset} conf={confidence:.2f}{p_def_str}{p_ema_str}{smooth_str}{vib_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Онлайн-детекция поломки винта (Random Forest)")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--deformed-threshold",
        type=float,
        default=DEFAULT_DEFORMED_THRESHOLD,
        help="Порог p_ema(DEFORMED) для вывода DEFORMED (по умолчанию 0.60)",
    )
    parser.add_argument(
        "--session-threshold",
        type=float,
        default=DEFAULT_SESSION_THRESHOLD,
        help="Порог доли DEFORMED окон для итогового вердикта (по умолчанию 0.20)",
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=DEFAULT_PEAK_THRESHOLD,
        help="Порог q90 по p_ema: выше => BROKEN даже при пограничной доле (по умолчанию 0.58)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=DEFAULT_EMA_ALPHA,
        help="Скорость сглаживания p_deformed (0..1, по умолчанию 0.25)",
    )
    parser.add_argument(
        "--invert-labels",
        action="store_true",
        help="Инвертировать p_deformed, если модель предсказывает классы наоборот",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    model_path = base / "propeller_fault_rf.pkl"
    if not model_path.exists():
        print(f"Модель не найдена: {model_path}", file=sys.stderr)
        print("Сначала: python3 train_rf_detector.py", file=sys.stderr)
        sys.exit(1)

    bundle = load_model(model_path)
    detector = RealtimeRFDetector(
        bundle,
        deformed_threshold=args.deformed_threshold,
        session_threshold=args.session_threshold,
        peak_threshold=args.peak_threshold,
        ema_alpha=args.ema_alpha,
        invert_labels=args.invert_labels,
    )
    model_type = bundle.get("model_type", "Unknown")
    print(f"Модель: {model_type} ({model_path.name})")
    print(f"Признаков: {len(bundle['feature_names'])}, окно: {WINDOW_SIZE}")
    print(f"Порог DEFORMED: p_ema >= {args.deformed_threshold:.2f}")
    print(
        f"Пороги сессии: ratio >= {args.session_threshold:.2f} или q90(p_ema) >= {args.peak_threshold:.2f}; "
        f"ema_alpha={args.ema_alpha:.2f}"
    )
    if args.invert_labels:
        print("Режим: INVERT_LABELS включён (инверсия p_deformed)")
    print("=" * 60)

    serial = SerialHandler(args.port, args.baud)

    serial.send_with_checksum(bytes.fromhex("FA FF 30 00"))
    print("Config mode...")
    time.sleep(0.1)

    serial.send_with_checksum(bytes.fromhex(
        "FA FF C0 20 10 20 FF FF 10 60 FF FF "
        "20 30 00 64 40 20 00 64 40 30 00 64 "
        "80 20 00 64 C0 20 00 64 E0 20 FF FF"
    ))
    print("Output configured (MTi-30, 100 Hz)...")
    time.sleep(0.1)

    serial.send_with_checksum(bytes.fromhex("FA FF 10 00"))
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
        p_def = detector.prediction_log[-1][3] if detector.prediction_log else None
        p_ema = detector.prediction_log[-1][4] if detector.prediction_log else None
        display_status(pred, conf, detector.smoothed_label, last_total[0], p_deformed=p_def, p_deformed_ema=p_ema)

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
            p_ema_values = [row[4] for row in detector.prediction_log]
            q90 = float(np.quantile(p_ema_values, 0.9)) if p_ema_values else 0.0
            broken = ratio >= detector.session_threshold or q90 >= detector.peak_threshold
            verdict = "BROKEN" if broken else "NORMAL"
            print(
                f"Итог для оператора: {verdict} "
                f"(ratio={ratio:.2f}, q90(p_ema)={q90:.2f})"
            )

            log_dir = base / "detection_logs"
            log_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"rf_detection_{ts}.csv"
            with open(log_path, "w") as f:
                f.write("elapsed_s,prediction,confidence,p_deformed,p_deformed_ema\n")
                for t, p, c, p_def, p_ema in detector.prediction_log:
                    f.write(f"{t:.3f},{p},{c:.4f},{p_def:.4f},{p_ema:.4f}\n")
            print(f"Лог: {log_path}")


if __name__ == "__main__":
    main()
