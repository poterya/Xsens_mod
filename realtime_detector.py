#!/usr/bin/env python3
"""
Детекция состояния винтов квадрокоптера в реальном времени через Xsens.

Загружает обученное дерево (propeller_fault_model.pkl), копит окно вибрации
и каждые STEP отсчётов выдаёт вердикт по классам:
  normal | front_left | front_right | rear_left | rear_right.

Условия:
- Если уверенность ниже CONFIDENCE_THRESHOLD (по умолчанию 0.65) — вывод не делается.
- В консоль печатается статус всех 4 винтов и общая уверенность.
- Время обработки одного пакета ограничено LATENCY_LIMIT_MS (50 мс).

По Ctrl+C: лог CSV + график предсказаний (PNG) в detection_logs/.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from DataPacketParser import DataPacketParser, XsDataPacket
from SerialHandler import SerialHandler
from XbusPacket import XbusPacket
from feature_extraction import extract_features

WINDOW_SIZE = 50
STEP = 10
CONFIDENCE_THRESHOLD = 0.65
LATENCY_LIMIT_MS = 50.0
SMOOTHING_WINDOW = 7  # размер окна сглаживания (мажоритарное голосование)

PROPELLER_LABELS = ("front_left", "front_right", "rear_left", "rear_right")
PROP_RU = {
    "front_left": "перед-лев",
    "front_right": "перед-прав",
    "rear_left": "зад-лев",
    "rear_right": "зад-прав",
}


def load_model(model_path: Path) -> dict:
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    required = {"classifier", "feature_names", "window_size", "class_labels"}
    missing = required - bundle.keys()
    if missing:
        raise ValueError(f"Модель не содержит обязательных ключей: {missing}")
    return bundle


class RealtimeFaultDetector:
    def __init__(self, model_bundle: dict, window_size: int = WINDOW_SIZE):
        self.clf = model_bundle["classifier"]
        self.feature_names: list[str] = model_bundle["feature_names"]
        self.class_labels: list[str] = list(model_bundle["class_labels"])
        self.window_size = window_size

        self.acc_x: deque = deque(maxlen=window_size)
        self.acc_y: deque = deque(maxlen=window_size)
        self.acc_z: deque = deque(maxlen=window_size)

        self.vib_total: deque = deque(maxlen=window_size)
        self.vib_x: deque = deque(maxlen=window_size)
        self.vib_y: deque = deque(maxlen=window_size)
        self.vib_z: deque = deque(maxlen=window_size)

        self.sample_count = 0
        self.start_time = time.time()
        self.prediction_log: list[tuple[float, str, float]] = []
        self.smoothing_buf: deque = deque(maxlen=SMOOTHING_WINDOW)
        self.vibration_log: list[tuple[float, float, float, float, float]] = []

    def add_accel(self, ax: float, ay: float, az: float) -> None:
        self.acc_x.append(ax)
        self.acc_y.append(ay)
        self.acc_z.append(az)
        self.sample_count += 1

        if len(self.acc_x) >= self.window_size:
            ax_arr = np.fromiter(self.acc_x, dtype=float, count=self.window_size)
            ay_arr = np.fromiter(self.acc_y, dtype=float, count=self.window_size)
            az_arr = np.fromiter(self.acc_z, dtype=float, count=self.window_size)
            mean_x = ax_arr.mean()
            mean_y = ay_arr.mean()
            mean_z = az_arr.mean()
            vx = ax_arr - mean_x
            vy = ay_arr - mean_y
            vz = az_arr - mean_z
            rms_x = float(np.sqrt(np.mean(vx * vx)))
            rms_y = float(np.sqrt(np.mean(vy * vy)))
            rms_z = float(np.sqrt(np.mean(vz * vz)))
            total = float(np.sqrt(rms_x ** 2 + rms_y ** 2 + rms_z ** 2))
            self.vib_total.append(total)
            self.vib_x.append(rms_x)
            self.vib_y.append(rms_y)
            self.vib_z.append(rms_z)
            t_rel = time.time() - self.start_time
            self.vibration_log.append((t_rel, total, rms_x, rms_y, rms_z))

    def predict(self) -> tuple[str, float, dict[str, float]] | None:
        if len(self.vib_total) < self.window_size:
            return None
        feats = extract_features(
            np.fromiter(self.vib_total, dtype=float, count=self.window_size),
            np.fromiter(self.vib_x, dtype=float, count=self.window_size),
            np.fromiter(self.vib_y, dtype=float, count=self.window_size),
            np.fromiter(self.vib_z, dtype=float, count=self.window_size),
        )
        x_vec = np.array(
            [[feats.get(name, 0.0) for name in self.feature_names]], dtype=float
        )
        proba = self.clf.predict_proba(x_vec)[0]
        best_idx = int(np.argmax(proba))
        pred_label = self.class_labels[best_idx]
        confidence = float(proba[best_idx])
        proba_map = {
            self.class_labels[i]: float(p) for i, p in enumerate(proba)
        }
        elapsed = time.time() - self.start_time
        self.smoothing_buf.append((pred_label, confidence))
        self.prediction_log.append((elapsed, pred_label, confidence))
        return pred_label, confidence, proba_map

    def smoothed_prediction(self) -> tuple[str, float] | None:
        """Мажоритарное голосование по последним SMOOTHING_WINDOW предсказаниям.
        Каждое предсказание учитывается с весом своей уверенности.
        """
        if len(self.smoothing_buf) < self.smoothing_buf.maxlen:
            return None
        weights: dict[str, float] = {}
        counts: dict[str, int] = {}
        for lbl, c in self.smoothing_buf:
            weights[lbl] = weights.get(lbl, 0.0) + c
            counts[lbl] = counts.get(lbl, 0) + 1
        best_lbl = max(weights, key=lambda k: (counts[k], weights[k]))
        avg_conf = weights[best_lbl] / counts[best_lbl]
        return best_lbl, float(avg_conf)


def display_status(
    pred_label: str,
    confidence: float,
    proba_map: dict[str, float],
    rms_total: float | None,
    elapsed_ms: float,
) -> None:
    if pred_label == "normal":
        head_color = "\033[92m"
        head_text = (
            f"NORMAL (все винты исправны) "
            f"уверенность(сглаж.)={confidence:.2f}"
        )
    else:
        head_color = "\033[91m"
        ru = PROP_RU.get(pred_label, pred_label)
        head_text = (
            f"DEFORMED — поломка винта: {ru} ({pred_label}) "
            f"уверенность(сглаж.)={confidence:.2f}"
        )
    reset = "\033[0m"

    parts: list[str] = []
    p_normal = proba_map.get("normal", 0.0)
    for prop in PROPELLER_LABELS:
        p_def = proba_map.get(prop, 0.0)
        ok_score = max(p_normal, 1.0 - p_def)
        if pred_label == prop:
            label = "ПОЛОМКА"
            color = "\033[91m"
            score = p_def
        else:
            label = "OK"
            color = "\033[92m"
            score = max(ok_score, 1.0 - p_def)
        parts.append(
            f"{PROP_RU[prop]:>10s}: {color}{label:<8s}{reset} {score:.2f}"
        )

    vib_str = f" vib={rms_total:.4f}" if rms_total is not None else ""
    print(f"{head_color}[{head_text}]{reset}{vib_str}  ({elapsed_ms:.1f} ms)")
    print("  " + " | ".join(parts))


def plot_prediction_session(
    prediction_log: list[tuple[float, str, float]],
    class_labels: list[str],
    out_path: Path,
) -> None:
    """График класса и уверенности по времени; PNG на диск + показ окна при наличии DISPLAY."""
    if not prediction_log:
        return

    t = np.array([row[0] for row in prediction_log], dtype=float)
    labels = [row[1] for row in prediction_log]
    conf = np.array([row[2] for row in prediction_log], dtype=float)
    label_to_idx = {lbl: i for i, lbl in enumerate(class_labels)}
    y = np.array([label_to_idx[l] for l in labels], dtype=int)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    for i, lbl in enumerate(class_labels):
        mask = y == i
        if np.any(mask):
            ax1.scatter(
                t[mask],
                y[mask],
                color=f"C{i % 10}",
                label=lbl,
                s=22,
                alpha=0.85,
                edgecolors="none",
            )

    ax1.set_yticks(range(len(class_labels)))
    ax1.set_yticklabels(class_labels, fontsize=9)
    ax1.set_ylabel("класс")
    ax1.set_title("Предсказания по времени (сессия realtime_detector)")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, conf, color="0.2", linewidth=0.9, alpha=0.9, label="уверенность")
    ax2.axhline(
        CONFIDENCE_THRESHOLD,
        color="tab:red",
        linestyle="--",
        linewidth=1,
        label=f"порог {CONFIDENCE_THRESHOLD:.2f}",
    )
    ax2.set_ylabel("уверенность")
    ax2.set_xlabel("время, с")
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"График сохранён: {out_path}")

    if os.environ.get("DISPLAY") or sys.platform == "darwin":
        plt.show()
    else:
        print(
            "DISPLAY не задан — окно не открыто; откройте PNG вручную.",
            file=sys.stderr,
        )
    plt.close(fig)


def plot_vibration_session(
    vibration_log: list[tuple[float, float, float, float, float]],
    out_path: Path,
) -> None:
    """График вибрации (TOTAL и по осям) от времени, рядом с предсказаниями."""
    if not vibration_log:
        return
    t = np.array([row[0] for row in vibration_log])
    total = np.array([row[1] for row in vibration_log])
    rx = np.array([row[2] for row in vibration_log])
    ry = np.array([row[3] for row in vibration_log])
    rz = np.array([row[4] for row in vibration_log])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    ax1.plot(t, total, color="tab:blue", linewidth=1.2)
    ax1.set_ylabel("TOTAL, m/s²")
    ax1.set_title("Вибрация за сессию")
    ax1.grid(True, alpha=0.3)
    ax2.plot(t, rx, label="RMS X", color="tab:red", linewidth=1.0)
    ax2.plot(t, ry, label="RMS Y", color="tab:green", linewidth=1.0)
    ax2.plot(t, rz, label="RMS Z", color="tab:blue", linewidth=1.0)
    ax2.set_xlabel("время, с")
    ax2.set_ylabel("RMS по осям")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"График вибрации сохранён: {out_path}")


def save_session(
    detector: RealtimeFaultDetector, flight_dir: Path, latency_violations: int
) -> None:
    flight_dir.mkdir(parents=True, exist_ok=True)

    pred_csv = flight_dir / "predictions.csv"
    with open(pred_csv, "w") as f:
        f.write("elapsed_s,prediction,confidence\n")
        for t, p, c in detector.prediction_log:
            f.write(f"{t:.3f},{p},{c:.4f}\n")
    print(f"Лог предсказаний: {pred_csv}")

    vib_csv = flight_dir / "vibration_log.csv"
    with open(vib_csv, "w") as f:
        f.write("time_seconds,total_vibration,rms_x,rms_y,rms_z\n")
        for t, total, rx, ry, rz in detector.vibration_log:
            f.write(f"{t:.3f},{total:.6f},{rx:.6f},{ry:.6f},{rz:.6f}\n")
    print(f"Лог вибрации: {vib_csv}")

    plot_prediction_session(
        detector.prediction_log,
        detector.class_labels,
        flight_dir / "predictions.png",
    )
    plot_vibration_session(detector.vibration_log, flight_dir / "vibration_plot.png")

    info = flight_dir / "session_info.txt"
    preds = [p[1] for p in detector.prediction_log]
    counts = {lbl: preds.count(lbl) for lbl in detector.class_labels}
    top = max(counts.items(), key=lambda kv: kv[1]) if counts else ("-", 0)
    duration = detector.prediction_log[-1][0] if detector.prediction_log else 0.0
    with open(info, "w") as f:
        f.write(f"Started:   {datetime.fromtimestamp(detector.start_time)}\n")
        f.write(f"Duration:  {duration:.2f} s\n")
        f.write(f"Predictions total: {len(preds)}\n")
        for lbl, n in counts.items():
            f.write(f"  {lbl:12s} {n}\n")
        f.write(f"Dominant: {top[0]} ({top[1]} predictions)\n")
        f.write(f"Latency >50ms violations: {latency_violations}\n")
        if detector.vibration_log:
            totals = [row[1] for row in detector.vibration_log]
            f.write(
                f"Vibration TOTAL  min={min(totals):.4f}  "
                f"avg={sum(totals)/len(totals):.4f}  max={max(totals):.4f}\n"
            )
    print(f"Сводка сессии: {info}")


def main() -> None:
    base = Path(__file__).resolve().parent
    model_path = base / "propeller_fault_model.pkl"
    if not model_path.exists():
        print(f"Модель не найдена: {model_path}", file=sys.stderr)
        print("Сначала запустите: python3 train_detector.py", file=sys.stderr)
        sys.exit(1)

    bundle = load_model(model_path)
    detector = RealtimeFaultDetector(bundle)
    cv_acc = bundle.get("cv_accuracy", float("nan"))
    print(f"Модель загружена: {model_path.name}")
    print(
        f"Классы: {', '.join(bundle['class_labels'])} | "
        f"cv_accuracy={cv_acc:.4f} | окно={WINDOW_SIZE}, шаг={STEP}, "
        f"сглаживание={SMOOTHING_WINDOW}, "
        f"порог уверенности={CONFIDENCE_THRESHOLD:.2f}, лимит={LATENCY_LIMIT_MS:.0f} ms"
    )
    print("=" * 80)

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
    latency_violations = [0]

    def on_packet(raw_packet):
        t_pkt = time.perf_counter()
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
        elapsed_ms = (time.perf_counter() - t_pkt) * 1000.0
        if elapsed_ms > LATENCY_LIMIT_MS:
            latency_violations[0] += 1
            print(
                f"\033[93m[!] обработка пакета {elapsed_ms:.1f} мс > "
                f"{LATENCY_LIMIT_MS:.0f} мс\033[0m",
                file=sys.stderr,
            )
        if result is None:
            return
        pred_label, conf, proba_map = result

        smoothed = detector.smoothed_prediction()
        if smoothed is None:
            return
        s_label, s_conf = smoothed
        if s_conf < CONFIDENCE_THRESHOLD:
            return
        display_status(s_label, s_conf, proba_map, last_total[0], elapsed_ms)

    packet = XbusPacket(on_data_available=on_packet)

    try:
        while True:
            byte = serial.read_byte()
            if byte:
                packet.feed_byte(byte)
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Остановлено.")
        if detector.prediction_log:
            preds = [p[1] for p in detector.prediction_log]
            counts = {lbl: preds.count(lbl) for lbl in detector.class_labels}
            print("Распределение предсказаний за сессию:")
            for lbl, n in counts.items():
                print(f"  {lbl:12s} {n}")
            top = max(counts.items(), key=lambda kv: kv[1])
            print(f"Доминирующий класс: {top[0]} ({top[1]} предсказаний)")
            if latency_violations[0]:
                print(
                    f"Превышений лимита {LATENCY_LIMIT_MS:.0f} мс: "
                    f"{latency_violations[0]}"
                )

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            flight_dir = base / "detection_logs" / f"flight_{ts}"
            print(f"Папка полёта: {flight_dir}")
            save_session(detector, flight_dir, latency_violations[0])
        else:
            print("Нет предсказаний — данные не сохранены.")


if __name__ == "__main__":
    main()
