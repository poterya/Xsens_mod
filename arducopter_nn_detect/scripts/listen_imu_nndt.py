#!/usr/bin/env python3
"""
Listen to live IMU vibration over MAVLink and run the propeller-fault
MLP detector in real time. No CSV, no recompile - works against any
ArduPilot vehicle (SITL or real flight controller) that publishes the
standard MAVLink ``VIBRATION`` message.

This is the Python sibling of the in-firmware ``NN_DETECT`` flight mode.
It uses the exact same data source (``AP_InertialSensor::get_vibration_levels()``
- which is what populates the MAVLink ``VIBRATION.vibration_{x,y,z}``
fields - see libraries/GCS_MAVLink/GCS_Common.cpp ``send_vibration``)
and the exact same feature extractor and MLP weights (``.pt`` from
branch ``nir``), so its decisions match the on-board detector to within
floating-point noise.

Usage::

    pip3 install --user pymavlink torch numpy scipy
    cd Xsens_mod/arducopter_nn_detect
    # against SITL started by ./sitl.sh:
    python3 scripts/listen_imu_nndt.py --master tcp:127.0.0.1:5760
    # against a real autopilot over USB:
    python3 scripts/listen_imu_nndt.py --master /dev/ttyACM0 --baud 115200
    # against a UDP telemetry link:
    python3 scripts/listen_imu_nndt.py --master udpin:0.0.0.0:14550

The program prints once per ``--print-period`` seconds and on each
DEFORMED/RECOVERED transition.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import torch
from pymavlink import mavutil


HERE = Path(__file__).resolve().parent
MODEL_EXPORT = HERE.parent / "model_export"
sys.path.insert(0, str(MODEL_EXPORT))

from feature_extraction import extract_features          # noqa: E402
from export_mlp_to_cpp import MLPDetector, feature_name_order  # noqa: E402


# ---- Constants that mirror the C++ mode_nn_detect.cpp ----------------
WINDOW_SIZE = 50                 # samples accumulated before each inference
SAMPLE_RATE_HZ = 100.0           # training rate; we ask SITL for this rate
DEFORMED_THRESHOLD = 0.50        # same as NN_DEFORMED_THRESHOLD
PROBA_EMA_ALPHA = 0.25           # same as NN_PROBA_EMA_ALPHA


def request_vibration_stream(m: mavutil.mavfile, rate_hz: float, comp: int) -> None:
    """Tell the autopilot to send VIBRATION at ``rate_hz``.

    Equivalent to the SET_MESSAGE_INTERVAL trick used by
    ``scripts/auto_flight_nndt.py``. ArduPilot computes vibration levels
    at the full IMU loop rate (400 Hz on Copter), so 100 Hz is safe.
    """
    ml = mavutil.mavlink
    interval_us = int(1_000_000 / rate_hz)
    m.mav.command_long_send(
        m.target_system, comp,
        ml.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        float(ml.MAVLINK_MSG_ID_VIBRATION),
        float(interval_us),
        0, 0, 0, 0, 0,
    )


def load_model(pt_path: Path) -> MLPDetector:
    model = MLPDetector(in_features=80)
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_proba(model: MLPDetector, features_dict: dict) -> float:
    order = feature_name_order()
    vec = np.array([features_dict[name] for name in order], dtype=np.float32)
    with torch.no_grad():
        logit = model(torch.from_numpy(vec).unsqueeze(0)).item()
    return 1.0 / (1.0 + math.exp(-logit))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default="tcp:127.0.0.1:5760",
                    help="MAVLink endpoint (tcp:..., udpin:..., /dev/ttyACM0, ...)")
    ap.add_argument("--baud", type=int, default=115200,
                    help="Baud rate (serial endpoints only)")
    ap.add_argument("--rate-hz", type=float, default=SAMPLE_RATE_HZ,
                    help="VIBRATION sample rate to request from the autopilot")
    ap.add_argument("--pt", type=Path,
                    default=MODEL_EXPORT / "propeller_fault_mlp_keras_set05.pt",
                    help="PyTorch checkpoint to use for inference")
    ap.add_argument("--threshold", type=float, default=DEFORMED_THRESHOLD,
                    help="Probability threshold for the DEFORMED verdict")
    ap.add_argument("--ema-alpha", type=float, default=PROBA_EMA_ALPHA,
                    help="EMA smoothing factor for P(fault)")
    ap.add_argument("--window", type=int, default=WINDOW_SIZE,
                    help="Sliding-window length in samples (training value: 50)")
    ap.add_argument("--print-period", type=float, default=1.0,
                    help="Print the current probability at most every N seconds")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="Stop after N seconds (0 = run forever)")
    ap.add_argument("--csv-out", type=Path, default=None,
                    help="Append per-sample (time,total,rx,ry,rz,p) into this CSV")
    args = ap.parse_args()

    if not args.pt.is_file():
        print(f"error: model checkpoint not found: {args.pt}", file=sys.stderr)
        return 1

    print(f"loading model: {args.pt}", flush=True)
    model = load_model(args.pt)
    print(f"connecting: {args.master}", flush=True)
    m = mavutil.mavlink_connection(args.master, baud=args.baud)
    m.wait_heartbeat()
    print(f"heartbeat sys={m.target_system} comp={m.target_component}", flush=True)
    autopilot_comp = getattr(mavutil.mavlink, "MAV_COMP_ID_AUTOPILOT1", 1)
    command_comp = m.target_component if m.target_component else autopilot_comp
    request_vibration_stream(m, args.rate_hz, command_comp)
    print(f"requested VIBRATION @ {args.rate_hz:.0f} Hz, listening...", flush=True)

    win_total: Deque[float] = deque(maxlen=args.window)
    win_x: Deque[float] = deque(maxlen=args.window)
    win_y: Deque[float] = deque(maxlen=args.window)
    win_z: Deque[float] = deque(maxlen=args.window)

    p_ema: Optional[float] = None
    last_verdict_broken = False
    last_print_t = 0.0
    started = time.time()
    samples_seen = 0
    inferences = 0

    csv_fp = None
    if args.csv_out is not None:
        csv_fp = args.csv_out.open("w", buffering=1)
        csv_fp.write("time_s,total,rms_x,rms_y,rms_z,p_raw,p_ema\n")

    try:
        while True:
            if args.max_seconds and (time.time() - started) > args.max_seconds:
                print("--max-seconds reached, exiting", flush=True)
                return 0

            msg = m.recv_match(type="VIBRATION", blocking=True, timeout=2.0)
            if msg is None:
                # Nothing arrived for 2 s - either link died or the
                # autopilot stopped streaming. Re-issue the request.
                print("warn: no VIBRATION for 2 s, re-requesting stream",
                      flush=True)
                request_vibration_stream(m, args.rate_hz, command_comp)
                continue

            vx, vy, vz = float(msg.vibration_x), float(msg.vibration_y), float(msg.vibration_z)
            tot = math.sqrt(vx * vx + vy * vy + vz * vz)
            win_total.append(tot)
            win_x.append(vx)
            win_y.append(vy)
            win_z.append(vz)
            samples_seen += 1

            p_raw: Optional[float] = None
            if len(win_total) >= args.window:
                feats = extract_features(
                    np.fromiter(win_total, dtype=np.float64, count=args.window),
                    np.fromiter(win_x,     dtype=np.float64, count=args.window),
                    np.fromiter(win_y,     dtype=np.float64, count=args.window),
                    np.fromiter(win_z,     dtype=np.float64, count=args.window),
                )
                p_raw = predict_proba(model, feats)
                inferences += 1
                if p_ema is None:
                    p_ema = p_raw
                else:
                    p_ema = (args.ema_alpha * p_raw
                             + (1.0 - args.ema_alpha) * p_ema)

            if csv_fp is not None:
                csv_fp.write(
                    f"{time.time() - started:.3f},{tot:.6f},{vx:.6f},"
                    f"{vy:.6f},{vz:.6f},"
                    f"{p_raw if p_raw is not None else float('nan'):.6f},"
                    f"{p_ema if p_ema is not None else float('nan'):.6f}\n")

            now = time.time()
            if p_ema is not None:
                broken_now = p_ema >= args.threshold
                # Transitions are reported immediately, status lines once
                # per --print-period.
                if broken_now != last_verdict_broken:
                    tag = "BROKEN" if broken_now else "RECOVERED"
                    print(f"[{now - started:7.2f}s] NN_DETECT: {tag} "
                          f"p={p_ema:.3f} vx={vx:.3f} vy={vy:.3f} vz={vz:.3f}",
                          flush=True)
                    last_verdict_broken = broken_now
                    last_print_t = now
                elif now - last_print_t >= args.print_period:
                    tag = "BROKEN" if broken_now else "ok"
                    print(f"[{now - started:7.2f}s] NN_DETECT: {tag} "
                          f"p={p_ema:.3f} vx={vx:.3f} vy={vy:.3f} vz={vz:.3f} "
                          f"(samples={samples_seen}, infer={inferences})",
                          flush=True)
                    last_print_t = now
            else:
                if now - last_print_t >= args.print_period:
                    fill = len(win_total)
                    print(f"[{now - started:7.2f}s] NN_DETECT: filling window "
                          f"({fill}/{args.window}) vx={vx:.3f} vy={vy:.3f} "
                          f"vz={vz:.3f}", flush=True)
                    last_print_t = now
    except KeyboardInterrupt:
        print("\nstopped by user", flush=True)
        return 0
    finally:
        if csv_fp is not None:
            csv_fp.close()


if __name__ == "__main__":
    raise SystemExit(main())
