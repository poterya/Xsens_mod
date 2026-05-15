#!/usr/bin/env bash
# Второй терминал после sitl.sh: MAVLink-клиент (arm / takeoff / NNDT + STATUSTEXT).
#
# Использование:
#   ./mavlink.sh
#   ./mavlink.sh --master tcp:127.0.0.1:5770
#   ./mavlink.sh --gps-timeout 180 --listen 120
#
# Все аргументы передаются в scripts/auto_flight_nndt.py.
# Если аргументов нет — подставляется --gps-timeout 180.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ROOT}/scripts/auto_flight_nndt.py"

if [[ ! -f "${PY}" ]]; then
  echo "Ошибка: не найден ${PY}" >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  exec python3 "${PY}" --gps-timeout 180
fi
exec python3 "${PY}" "$@"
