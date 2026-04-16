#!/usr/bin/env python3
"""
Детекция поломки винта квадрокоптера в реальном времени через Xsens.

Загружает обученную модель (propeller_fault_model.pkl), подключается к Xsens
по serial, копит скользящее окно вибрации и каждые WINDOW_SIZE отсчётов
выдаёт вердикт: NORMAL / DEFORMED (поломка винта).
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

import numpy as np

from SerialHandler import SerialHandler
from XbusPacket import XbusPacket
from DataPacketParser import DataPacketParser, XsDataPacket


WINDOW_SIZE = 50
STEP = 10
CONFIDENCE_QUEUE_SIZE = 5
DEFAULT_DEFORMED_THRESHOLD = 0.60
DEFAULT_SESSION_THRESHOLD = 0.22
DEFAULT_EMA_ALPHA = 0.25
DEFAULT_PEAK_THRESHOLD = 0.65
DEFAULT_STREAK_THRESHOLD = 20
DEFAULT_WARMUP_SECONDS = 10.0


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
    def __init__(
        self,
        model_bundle: dict,
        window_size: int = WINDOW_SIZE,
        deformed_threshold: float = DEFAULT_DEFORMED_THRESHOLD,
        session_threshold: float = DEFAULT_SESSION_THRESHOLD,
        peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
        streak_threshold: int = DEFAULT_STREAK_THRESHOLD,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        invert_labels: bool = False,
    ):
        self.clf = model_bundle["classifier"]
        self.feature_names = model_bundle["feature_names"]
        self.window_size = window_size
        self.deformed_threshold = deformed_threshold
        self.session_threshold = session_threshold
        self.peak_threshold = peak_threshold
        self.streak_threshold = streak_threshold
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
    print(f"{color}[{label}]{reset} уверенность={confidence:.2f}{p_def_str}{p_ema_str}{smooth_str}{vib_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Онлайн-детекция поломки винта (Decision Tree)")
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
        help="Порог доли DEFORMED окон для итогового вердикта (по умолчанию 0.22)",
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=DEFAULT_PEAK_THRESHOLD,
        help="Порог q90 по p_ema для устойчивой деформации (по умолчанию 0.65)",
    )
    parser.add_argument(
        "--streak-threshold",
        type=int,
        default=DEFAULT_STREAK_THRESHOLD,
        help="Минимальная длина серии DEFORMED для устойчивого BROKEN (по умолчанию 20)",
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
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=DEFAULT_WARMUP_SECONDS,
        help="Сколько секунд после старта не выполнять детекцию (по умолчанию 10)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    model_path = base / "propeller_fault_model.pkl"
    if not model_path.exists():
        print(f"Модель не найдена: {model_path}", file=sys.stderr)
        print("Сначала запустите: python3 train_detector.py", file=sys.stderr)
        sys.exit(1)

    bundle = load_model(model_path)
    detector = RealtimeFaultDetector(
        bundle,
        deformed_threshold=args.deformed_threshold,
        session_threshold=args.session_threshold,
        peak_threshold=args.peak_threshold,
        streak_threshold=args.streak_threshold,
        ema_alpha=args.ema_alpha,
        invert_labels=args.invert_labels,
    )
    print(f"Модель загружена: {model_path.name}")
    print(f"Окно: {WINDOW_SIZE}, шаг предсказания: {STEP}, сглаживание: {CONFIDENCE_QUEUE_SIZE}")
    print(f"Порог DEFORMED: p_ema >= {args.deformed_threshold:.2f}")
    print(
        f"Пороги сессии: ratio >= {args.session_threshold:.2f}, q90(p_ema) >= {args.peak_threshold:.2f}, "
        f"streak >= {args.streak_threshold}; ema_alpha={args.ema_alpha:.2f}"
    )
    print(f"Прогрев: первые {args.warmup_seconds:.1f} сек без детекции")
    if args.invert_labels:
        print("Режим: INVERT_LABELS включён (инверсия p_deformed)")
    print("=" * 60)

    serial = SerialHandler(args.port, args.baud)

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
    warmup_start = time.time()
    warmup_done = [False]

    def on_packet(raw_packet):
        xbus_data = XsDataPacket()
        DataPacketParser.parse_data_packet(raw_packet, xbus_data)

        if not xbus_data.accAvailable:
            return

        detector.add_accel(xbus_data.acc[0], xbus_data.acc[1], xbus_data.acc[2])

        if len(detector.vib_total) > 0:
            last_total[0] = float(detector.vib_total[-1])

        elapsed_warmup = time.time() - warmup_start
        if elapsed_warmup < args.warmup_seconds:
            return
        if not warmup_done[0]:
            warmup_done[0] = True
            print(f"\nПрогрев завершён ({args.warmup_seconds:.1f} сек). Детекция запущена.\n")

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
            max_streak = 0
            cur = 0
            for p in preds:
                if p == 1:
                    cur += 1
                    if cur > max_streak:
                        max_streak = cur
                else:
                    cur = 0

            broken = (
                (ratio >= detector.session_threshold and max_streak >= detector.streak_threshold // 2)
                or (q90 >= detector.peak_threshold and max_streak >= detector.streak_threshold)
            )
            verdict = "BROKEN" if broken else "NORMAL"
            print(
                f"Итог для оператора: {verdict} "
                f"(ratio={ratio:.2f}, q90(p_ema)={q90:.2f}, max_streak={max_streak})"
            )

            log_dir = base / "detection_logs"
            log_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"detection_{ts}.csv"
            with open(log_path, "w") as f:
                f.write("elapsed_s,prediction,confidence,p_deformed,p_deformed_ema\n")
                for t, p, c, p_def, p_ema in detector.prediction_log:
                    f.write(f"{t:.3f},{p},{c:.4f},{p_def:.4f},{p_ema:.4f}\n")
            print(f"Лог сохранён: {log_path}")


if __name__ == "__main__":
    main()
