#!/usr/bin/env python3
"""
Автоматический сценарий для варианта B (hover-gated CSV):
  GUIDED -> arm -> takeoff -> mode NNDT (29) -> печать NN_DETECT STATUSTEXT.

Первый терминал: start_sitl_csv_hover.sh <имя_файла.csv>
Второй терминал:  auto_flight_nndt.py

Требуется: pip install pymavlink
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from pymavlink import mavutil


def _param_id(name: str) -> bytes:
    b = name.encode("ascii")[:16]
    return b + bytes(max(0, 16 - len(b)))


def _set_param_scalar(
    m, comp: int, name: str, value: float, p_type: int
) -> None:
    m.mav.param_set_send(
        m.target_system,
        comp,
        _param_id(name),
        float(value),
        p_type,
    )


def _sitl_request_gps_streams(m, comp: int) -> None:
    """Попросить ArduPilot слать GPS/позицию чаще (через MAVLink2)."""
    ml = mavutil.mavlink
    for msg_id, hz in [
        (ml.MAVLINK_MSG_ID_GPS_RAW_INT, 5.0),
        (ml.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 5.0),
    ]:
        interval_us = int(1_000_000 / hz)
        m.mav.command_long_send(
            m.target_system,
            comp,
            ml.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            float(msg_id),
            float(interval_us),
            0,
            0,
            0,
            0,
            0,
        )


def wait_gps_ready(
    m,
    comp: int,
    min_fix_type: int = 3,
    timeout_sec: float = 120.0,
) -> bool:
    """
    Ждём 3D-fix по GPS_RAW_INT. Раньше скрипт слушал только этот тип и мог
    «висеть», если коптер почти не шлёт сообщение через TCP без
    MAV_CMD_SET_MESSAGE_INTERVAL.
    """
    _sitl_request_gps_streams(m, comp)
    deadline = time.time() + timeout_sec
    last_fix: Optional[int] = None
    last_print = 0.0
    print(f"Ждём GPS fix (>={min_fix_type}), до {timeout_sec:.0f} с...", flush=True)
    while time.time() < deadline:
        msg = m.recv_match(blocking=True, timeout=1.0)
        if msg is None:
            continue
        if msg.get_type() == "GPS_RAW_INT":
            last_fix = msg.fix_type
            if msg.fix_type >= min_fix_type:
                print(f"GPS fix_type={msg.fix_type}", flush=True)
                return True
        now = time.time()
        if now - last_print > 5.0:
            print(
                f"  … пока без 3D-fix (последний fix_type из GPS_RAW_INT={last_fix})",
                flush=True,
            )
            last_print = now
    print(
        "timeout: не дождались GPS 3D-fix. Проверь: один клиент на tcp:5760, "
        "порт не занят, SITL не упал; при необходимости увеличь --gps-timeout.",
        file=sys.stderr,
    )
    return False


def wait_armed_and_ack(
    m,
    arm_cmd_id: int,
    timeout_sec: float = 25.0,
) -> tuple[bool, Optional[int]]:
    """armed по HEARTBEAT + необязательный COMMAND_ACK MAV_CMD_COMPONENT_ARM_DISARM."""
    ml = mavutil.mavlink
    flag = getattr(ml, "MAV_MODE_FLAG_SAFETY_ARMED", 128)
    deadline = time.time() + timeout_sec
    armed = False
    ack_result: Optional[int] = None
    while time.time() < deadline:
        msg = m.recv_match(blocking=True, timeout=0.5)
        if msg is None:
            continue
        typ = msg.get_type()
        if typ == "HEARTBEAT" and int(msg.base_mode) & int(flag):
            armed = True
            break
        if typ == "COMMAND_ACK" and int(msg.command) == int(arm_cmd_id):
            ack_result = int(msg.result)
    return armed, ack_result


def wait_command_ack(
    m,
    expected_cmd: int,
    timeout_sec: float = 10.0,
) -> tuple[bool, Optional[int]]:
    """Возвращает (успех, result) по COMMAND_ACK.expected_cmd."""
    deadline = time.time() + timeout_sec
    ml = mavutil.mavlink
    while time.time() < deadline:
        ack = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
        if ack is None:
            continue
        if getattr(ack, "command", -1) == expected_cmd:
            ok = ack.result == ml.MAV_RESULT_ACCEPTED
            return ok, int(ack.result)
    return False, None


def main() -> int:
    p = argparse.ArgumentParser(description="SITL: arm, takeoff, mode NNDT, print NN_DETECT status")
    p.add_argument(
        "--master",
        default="tcp:127.0.0.1:5760",
        help="MAVLink master (по умолчанию tcp:127.0.0.1:5760 для SITL -I0)",
    )
    p.add_argument(
        "--takeoff-alt",
        type=float,
        default=5.0,
        help="Высота takeoff в метрах",
    )
    p.add_argument(
        "--listen",
        type=float,
        default=120.0,
        help="Сколько секунд слушать STATUSTEXT после включения NNDT",
    )
    p.add_argument(
        "--gps-timeout",
        type=float,
        default=120.0,
        help="Таймаут ожидания GPS 3D-fix (сек)",
    )
    p.add_argument(
        "--ekf-delay",
        type=float,
        default=5.0,
        help="Пауза после GPS перед командами (EKF успокоится)",
    )
    p.add_argument(
        "--alt-timeout",
        type=float,
        default=90.0,
        help="Таймаут набора высоты после NAV_TAKEOFF (сек)",
    )
    p.add_argument(
        "--no-relax-prearm",
        dest="relax_prearm",
        action="store_false",
        default=True,
        help="Не отключать prearm (не менять ARMING_CHECK в SITL)",
    )
    args = p.parse_args()

    print(f"Подключение: {args.master}", flush=True)
    m = mavutil.mavlink_connection(args.master)
    m.wait_heartbeat()
    print(f"heartbeat sys={m.target_system} comp={m.target_component}", flush=True)
    # Heartbeat может прийти с comp=0; для command_long надёжнее AUTOPILOT1.
    autopilot_comp = getattr(
        mavutil.mavlink,
        "MAV_COMP_ID_AUTOPILOT1",
        1,
    )
    command_comp = m.target_component if m.target_component else autopilot_comp

    if not wait_gps_ready(m, command_comp, 3, args.gps_timeout):
        return 1

    ml = mavutil.mavlink

    # Дать немного времени EKF/GPS фильтрам (иначе арм/takeoff иногда режутся)
    if args.ekf_delay > 0:
        print(f"Пауза {args.ekf_delay:.0f} с перед командами (EKF)...", flush=True)
        time.sleep(args.ekf_delay)

    if args.relax_prearm:
        print("SITL: ARMING_CHECK=0 (relax prearm)...", flush=True)
        _set_param_scalar(
            m,
            command_comp,
            "ARMING_CHECK",
            0.0,
            ml.MAV_PARAM_TYPE_REAL32,
        )
        time.sleep(0.5)

    def set_mode_custom(mode_num: int) -> None:
        # base_mode MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
        m.mav.command_long_send(
            m.target_system,
            command_comp,
            ml.MAV_CMD_DO_SET_MODE,
            0,
            1,
            float(mode_num),
            0,
            0,
            0,
            0,
            0,
        )

    print("mode GUIDED (4)...", flush=True)
    set_mode_custom(4)
    time.sleep(1.5)

    print("arm...", flush=True)
    m.mav.command_long_send(
        m.target_system,
        command_comp,
        ml.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    armed_ok, ack_arm_res = wait_armed_and_ack(
        m,
        int(ml.MAV_CMD_COMPONENT_ARM_DISARM),
        25.0,
    )
    print(
        f"arm: heartbeat_armed={armed_ok} COMMAND_ACK_result={ack_arm_res}",
        flush=True,
    )
    if not armed_ok:
        print("error: коптер не взвёлся за 25 с (проверь prearm / STATUSTEXT)", file=sys.stderr)
        return 2

    time.sleep(1.0)

    print(f"takeoff {args.takeoff_alt} m...", flush=True)
    m.mav.command_long_send(
        m.target_system,
        command_comp,
        ml.MAV_CMD_NAV_TAKEOFF,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        float(args.takeoff_alt),
    )
    ack_to_ok, ack_to_res = wait_command_ack(m, int(ml.MAV_CMD_NAV_TAKEOFF))
    print(
        f"takeoff COMMAND_ACK={ack_to_res} ({'ok' if ack_to_ok else 'ignored/unknown'})",
        flush=True,
    )

    tgt = args.takeoff_alt * 0.9
    print(f"Ждём высоту > {tgt:.1f} m (до {args.alt_timeout:.0f} с)...", flush=True)
    deadline = time.time() + args.alt_timeout
    while time.time() < deadline:
        pos = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0)
        if pos is None:
            continue
        rel_m = pos.relative_alt / 1000.0
        if rel_m > tgt:
            print(f"alt relative={rel_m:.1f} m", flush=True)
            break
    else:
        print("timeout: не достигли высоты", file=sys.stderr)
        return 1

    print("mode NNDT (29)...", flush=True)
    set_mode_custom(29)

    print("Ждём hover-gate + NN_DETECT STATUSTEXT (Ctrl+C чтобы выйти)...", flush=True)
    end = time.time() + args.listen
    while time.time() < end:
        msg = m.recv_match(blocking=True, timeout=0.5)
        if msg is None:
            continue
        if msg.get_type() == "STATUSTEXT" and "NN_DETECT" in msg.text:
            print(msg.text, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
