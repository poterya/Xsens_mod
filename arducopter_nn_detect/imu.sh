#!/usr/bin/env bash
# Live IMU listener: подключается по MAVLink к запущенному автопилоту
# (SITL или реальная плата), читает VIBRATION на 100 Гц и крутит ту же
# MLP-модель, что и встроенный режим NN_DETECT, прямо в Python.
#
# Использование:
#   ./imu.sh                                # SITL по умолчанию (tcp:127.0.0.1:5760)
#   ./imu.sh --master udpin:0.0.0.0:14550   # GCS-стиль через UDP
#   ./imu.sh --master /dev/ttyACM0 --baud 115200   # реальная плата
#   ./imu.sh --csv-out /tmp/live_vib.csv    # параллельно писать лог
#   ./imu.sh --max-seconds 30               # ограничить время прослушки
#


set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ROOT}/scripts/listen_imu_nndt.py"

if [[ ! -f "${PY}" ]]; then
  echo "ошибка: не найден ${PY}" >&2
  exit 1
fi

exec python3 "${PY}" "$@"
