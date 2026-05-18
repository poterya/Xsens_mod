#!/usr/bin/env python3
"""
Бинарная детекция поломки винта (NORMAL / FAULT) на основе фильтра Калмана.

Идея:
- Состояние x = [total, rms_x, rms_y, rms_z] — медленно меняющийся базовый
  уровень вибрации, оцениваемый Калманом.
- Innovation y_k = z_k - x_pred_k, ковариация S_k = P_pred_k + R.
- Тестовая статистика — квадрат расстояния Махаланобиса d²_k = y_k^T S_k^-1 y_k
  (под нормой ~ χ², df=4).
- Для устойчивости считаем EWMA от d² и берём p_fault = chi2.cdf(EWMA d², df=4).
  Поломка даёт ПОСТОЯННОЕ отклонение → EWMA «уезжает» вверх и p_fault → 1.

Калибровка:
- μ и R вычисляются по сессиям Normal_mod указанного датасета (default set_03).
- Q = R * Q_FACTOR — небольшое «расползание» базового уровня на каждый шаг,
  чтобы трекать естественные дрейфы (батарея, температура), но не быстро
  перехватывать резкие изменения от поломки.

Запуск:
    python3 kalman_fault_detector.py                     # калибровка из datasets/set_03
    python3 kalman_fault_detector.py --base datasets/set_02
    python3 kalman_fault_detector.py --offline path/to/vibration_log.csv

При --offline данные читаются из CSV и НЕ требуется Xsens — удобно для проверки.

По Ctrl+C/завершении: detection_logs/kalman_flight_<ts>/ с логами и графиками.
"""
from __future__ import annotations

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

# Параметры по умолчанию ----------------------------------------------------
WINDOW_SIZE = 50              # окно для расчёта RMS (как в realtime_detector)
LATENCY_LIMIT_MS = 50.0
CONFIDENCE_THRESHOLD = 0.65   # ниже не выводим в консоль
EWMA_ALPHA = 0.05             # сглаживание d² (≈ 1 / (1-α) ≈ 20 семплов)
Q_FACTOR = 1e-3               # Q = R * Q_FACTOR — слабый процессный шум
P0_FACTOR = 0.1               # начальная P = R * P0_FACTOR
DOF = 4                       # размерность измерения z

CHI2_99 = chi2.ppf(0.99, DOF)         # ~13.28
CHI2_999 = chi2.ppf(0.999, DOF)       # ~18.47


def calibrate_from_normal(normal_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """μ и R оцениваются из всех vibration_log.csv в Normal_mod.
    Берётся стационарная часть сессии (после первых WINDOW_SIZE строк).
    """
    rows: list[np.ndarray] = []
    for csv in sorted(normal_dir.rglob("vibration_log.csv")):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        cols = ["total_vibration", "rms_x", "rms_y", "rms_z"]
        if not set(cols).issubset(df.columns):
            continue
        z = df[cols].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
        if len(z) > WINDOW_SIZE * 2:
            z = z[WINDOW_SIZE:]  # отбрасываем начальный «прогрев»
        if len(z) > 0:
            rows.append(z)
    if not rows:
        raise SystemExit(f"Не найдено данных Normal в {normal_dir}")
    Z = np.vstack(rows)
    mu = Z.mean(axis=0)
    R = np.cov(Z, rowvar=False)
    # подстраховка от вырождения
    R = R + np.eye(DOF) * (np.trace(R) / DOF) * 1e-3
    return mu, R


# Калман-детектор -----------------------------------------------------------


class KalmanFaultDetector:
    """4-канальный (z = [total, rms_x, rms_y, rms_z]) бесшумный по динамике
    фильтр Калмана с тестом на innovation (Mahalanobis / chi²).
    Модель: x_{k+1} = x_k + w_k, z_k = x_k + v_k.
    """

    def __init__(
        self,
        mu0: np.ndarray,
        R: np.ndarray,
        q_factor: float = Q_FACTOR,
        p0_factor: float = P0_FACTOR,
        ewma_alpha: float = EWMA_ALPHA,
    ) -> None:
        self.x = np.asarray(mu0, dtype=float).copy()
        self.R = np.asarray(R, dtype=float).copy()
        self.Q = self.R * q_factor
        self.P = self.R * p0_factor
        self.alpha = float(ewma_alpha)
        self.ewma_d2 = 0.0
        self.start_time = time.time()
        # Лог состояния для отчёта
        self.state_log: list[tuple[float, np.ndarray, np.ndarray, float, float, float]] = []

    def update(self, z: np.ndarray) -> dict:
        z = np.asarray(z, dtype=float)
        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q
        # Innovation
        y = z - x_pred
        S = P_pred + self.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = P_pred @ S_inv
        self.x = x_pred + K @ y
        self.P = (np.eye(DOF) - K) @ P_pred

        d2 = float(y.T @ S_inv @ y)
        self.ewma_d2 = self.alpha * d2 + (1.0 - self.alpha) * self.ewma_d2
        p_fault = float(chi2.cdf(self.ewma_d2, DOF))

        t_rel = time.time() - self.start_time
        self.state_log.append((t_rel, z.copy(), self.x.copy(), float(d2), float(self.ewma_d2), float(p_fault)))
        return {
            "innovation": y,
            "d2": d2,
            "ewma_d2": self.ewma_d2,
            "p_fault": p_fault,
        }


# Утилита: онлайн-RMS (как в realtime_detector) ------------------------------


class _RMSWindow:
    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        self.n = window_size
        self.bx: deque = deque(maxlen=window_size)
        self.by: deque = deque(maxlen=window_size)
        self.bz: deque = deque(maxlen=window_size)

    def push(self, ax: float, ay: float, az: float) -> tuple[float, float, float, float] | None:
        self.bx.append(ax)
        self.by.append(ay)
        self.bz.append(az)
        if len(self.bx) < self.n:
            return None
        x = np.fromiter(self.bx, float, self.n)
        y = np.fromiter(self.by, float, self.n)
        z = np.fromiter(self.bz, float, self.n)
        vx = x - x.mean()
        vy = y - y.mean()
        vz = z - z.mean()
        rx = float(np.sqrt(np.mean(vx * vx)))
        ry = float(np.sqrt(np.mean(vy * vy)))
        rz = float(np.sqrt(np.mean(vz * vz)))
        total = float(np.sqrt(rx * rx + ry * ry + rz * rz))
        return total, rx, ry, rz


# Вывод ---------------------------------------------------------------------


def display_status(p_fault: float, ewma_d2: float, total_vib: float, elapsed_ms: float) -> None:
    is_fault = p_fault >= 0.5
    display_conf = p_fault if is_fault else (1.0 - p_fault)
    if display_conf < CONFIDENCE_THRESHOLD:
        return
    if is_fault:
        color = "\033[91m"
        head = (
            f"FAULT — поломка винта "
            f"уверенность={display_conf:.2f}  EWMA d²={ewma_d2:.2f} (χ²₉₉={CHI2_99:.2f})"
        )
    else:
        color = "\033[92m"
        head = (
            f"NORMAL уверенность={display_conf:.2f}  "
            f"EWMA d²={ewma_d2:.2f} (χ²₉₉={CHI2_99:.2f})"
        )
    reset = "\033[0m"
    print(f"{color}[{head}]{reset}  vib={total_vib:.4f}  ({elapsed_ms:.1f} ms)")


# Сохранение «полёта» -------------------------------------------------------


def save_flight(
    detector: KalmanFaultDetector,
    flight_dir: Path,
    latency_violations: int = 0,
) -> None:
    flight_dir.mkdir(parents=True, exist_ok=True)
    log_csv = flight_dir / "kalman_log.csv"
    with open(log_csv, "w") as f:
        f.write(
            "time_seconds,total,rms_x,rms_y,rms_z,"
            "baseline_total,baseline_x,baseline_y,baseline_z,"
            "d2,ewma_d2,p_fault,decision\n"
        )
        for t, z, x, d2, ewma, pf in detector.state_log:
            decision = "FAULT" if pf >= 0.5 else "NORMAL"
            f.write(
                f"{t:.3f},{z[0]:.6f},{z[1]:.6f},{z[2]:.6f},{z[3]:.6f},"
                f"{x[0]:.6f},{x[1]:.6f},{x[2]:.6f},{x[3]:.6f},"
                f"{d2:.4f},{ewma:.4f},{pf:.4f},{decision}\n"
            )
    print(f"Лог Калмана: {log_csv}")

    vib_csv = flight_dir / "vibration_log.csv"
    with open(vib_csv, "w") as f:
        f.write("time_seconds,total_vibration,rms_x,rms_y,rms_z\n")
        for t, z, _x, _d2, _ewma, _pf in detector.state_log:
            f.write(f"{t:.3f},{z[0]:.6f},{z[1]:.6f},{z[2]:.6f},{z[3]:.6f}\n")
    print(f"Лог вибрации: {vib_csv}")

    plot_path = flight_dir / "kalman_plot.png"
    plot_kalman_session(detector, plot_path)

    info = flight_dir / "session_info.txt"
    if detector.state_log:
        duration = detector.state_log[-1][0]
        pfaults = [row[5] for row in detector.state_log]
        n_fault = sum(1 for p in pfaults if p >= 0.5)
        n_total = len(pfaults)
        with open(info, "w") as f:
            f.write(f"Started:   {datetime.fromtimestamp(detector.start_time)}\n")
            f.write(f"Duration:  {duration:.2f} s\n")
            f.write(f"Samples:   {n_total}\n")
            f.write(f"Fault:     {n_fault}  ({100*n_fault/max(n_total,1):.2f}%)\n")
            f.write(f"Normal:    {n_total - n_fault}\n")
            avg_pfault = float(np.mean(pfaults)) if pfaults else 0.0
            f.write(f"Avg p_fault:  {avg_pfault:.4f}\n")
            f.write(f"Latency >50ms violations: {latency_violations}\n")
        print(f"Сводка сессии: {info}")


def plot_kalman_session(detector: KalmanFaultDetector, out_path: Path) -> None:
    if not detector.state_log:
        return
    t = np.array([r[0] for r in detector.state_log])
    z = np.vstack([r[1] for r in detector.state_log])
    x = np.vstack([r[2] for r in detector.state_log])
    d2 = np.array([r[3] for r in detector.state_log])
    ewma = np.array([r[4] for r in detector.state_log])
    pf = np.array([r[5] for r in detector.state_log])

    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)
    ax1, ax2, ax3 = axes

    ax1.plot(t, z[:, 0], label="TOTAL (измерено)", color="0.2", linewidth=1.0)
    ax1.plot(t, x[:, 0], label="базовый уровень (Калман)", color="tab:blue", linewidth=1.2)
    ax1.set_ylabel("TOTAL, m/s²")
    ax1.set_title("Калмановский детектор поломки винта")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(t, np.maximum(d2, 1e-3), color="0.5", linewidth=0.7, alpha=0.7, label="d² мгновенный")
    ax2.semilogy(t, np.maximum(ewma, 1e-3), color="tab:red", linewidth=1.4, label="EWMA d²")
    ax2.axhline(CHI2_99, color="tab:orange", linestyle="--", linewidth=1, label=f"χ²₉₉≈{CHI2_99:.2f}")
    ax2.axhline(CHI2_999, color="tab:purple", linestyle="--", linewidth=1, label=f"χ²₉₉₉≈{CHI2_999:.2f}")
    ax2.set_ylabel("статистика, лог-шкала")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which="both")

    ax3.plot(t, pf, color="tab:red", linewidth=1.0, label="p_fault")
    ax3.axhline(0.5, color="0.4", linestyle="--", linewidth=0.8, label="порог 0.5")
    ax3.fill_between(t, 0, 1, where=pf >= 0.5, color="tab:red", alpha=0.10)
    ax3.set_xlabel("время, с")
    ax3.set_ylabel("p_fault")
    ax3.set_ylim(-0.02, 1.02)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"График сохранён: {out_path}")
    if os.environ.get("DISPLAY") or sys.platform == "darwin":
        plt.show()
    plt.close(fig)


# Точки входа ---------------------------------------------------------------


def run_offline(detector: KalmanFaultDetector, csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    cols = ["total_vibration", "rms_x", "rms_y", "rms_z"]
    Z = df[cols].to_numpy()
    times = (
        df["time_seconds"].to_numpy()
        if "time_seconds" in df.columns
        else np.arange(len(Z)) * 0.01
    )
    n_fault = 0
    for t, z in zip(times, Z):
        out = detector.update(z)
        if out["p_fault"] >= 0.5:
            n_fault += 1
        # Показываем кратко: каждые 50 строк
    print(
        f"Offline: rows={len(Z)}, FAULT-семплов={n_fault} "
        f"({100*n_fault/max(len(Z),1):.1f}%), avg p_fault="
        f"{np.mean([row[5] for row in detector.state_log]):.4f}"
    )


def run_realtime(detector: KalmanFaultDetector, base_dir: Path) -> None:
    # Импорт Xsens только в realtime, чтобы offline-режим работал и без serial
    from xsens.DataPacketParser import DataPacketParser, XsDataPacket
    from xsens.SerialHandler import SerialHandler
    from xsens.XbusPacket import XbusPacket

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

    rms_win = _RMSWindow(WINDOW_SIZE)
    last_total = [None]
    latency_violations = [0]

    def on_packet(raw_packet):
        t0 = time.perf_counter()
        xbus = XsDataPacket()
        DataPacketParser.parse_data_packet(raw_packet, xbus)
        if not xbus.accAvailable:
            return
        rms = rms_win.push(xbus.acc[0], xbus.acc[1], xbus.acc[2])
        if rms is None:
            return
        total, rx, ry, rz = rms
        z = np.array([total, rx, ry, rz], dtype=float)
        out = detector.update(z)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > LATENCY_LIMIT_MS:
            latency_violations[0] += 1
            print(
                f"\033[93m[!] обработка пакета {elapsed_ms:.1f} мс > "
                f"{LATENCY_LIMIT_MS:.0f} мс\033[0m",
                file=sys.stderr,
            )
        last_total[0] = total
        display_status(out["p_fault"], out["ewma_d2"], total, elapsed_ms)

    packet = XbusPacket(on_data_available=on_packet)
    try:
        while True:
            byte = serial.read_byte()
            if byte:
                packet.feed_byte(byte)
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Остановлено.")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        flight_dir = base_dir / "detection_logs" / f"kalman_flight_{ts}"
        save_flight(detector, flight_dir, latency_violations[0])


def main() -> None:
    here = _PROJECT_ROOT
    parser = argparse.ArgumentParser(
        description="Kalman-детектор поломки винта (бинарно: NORMAL / FAULT)"
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=here / "datasets" / "set_03",
        help="Каталог с Normal_mod (для калибровки μ и R)",
    )
    parser.add_argument(
        "--offline",
        type=Path,
        default=None,
        help="Запустить на готовом vibration_log.csv (без Xsens)",
    )
    parser.add_argument("--ewma-alpha", type=float, default=EWMA_ALPHA)
    parser.add_argument("--q-factor", type=float, default=Q_FACTOR)
    args = parser.parse_args()

    normal_dir = args.base / "Normal_mod"
    if not normal_dir.is_dir():
        raise SystemExit(f"Нет {normal_dir}; укажите --base правильно")

    print(f"Калибровка по Normal: {normal_dir}")
    mu, R = calibrate_from_normal(normal_dir)
    print("μ =", np.round(mu, 4))
    print("diag(R) =", np.round(np.diag(R), 6))

    detector = KalmanFaultDetector(
        mu0=mu,
        R=R,
        q_factor=args.q_factor,
        p0_factor=P0_FACTOR,
        ewma_alpha=args.ewma_alpha,
    )

    print(
        f"Параметры: WINDOW={WINDOW_SIZE}, EWMA_ALPHA={args.ewma_alpha}, "
        f"Q_FACTOR={args.q_factor}, лимит={LATENCY_LIMIT_MS:.0f} ms, "
        f"порог уверенности={CONFIDENCE_THRESHOLD:.2f}"
    )
    print(f"Пороги: χ²₉₉≈{CHI2_99:.2f}, χ²₉₉₉≈{CHI2_999:.2f}")
    print("=" * 80)

    if args.offline is not None:
        run_offline(detector, args.offline)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        flight_dir = here / "detection_logs" / f"kalman_offline_{ts}"
        save_flight(detector, flight_dir, latency_violations=0)
    else:
        run_realtime(detector, here)


if __name__ == "__main__":
    main()
