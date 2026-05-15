#!/usr/bin/env bash
# Терминал 2: клиент после шага 1 — GUIDED/arm/takeoff/mode NNDT + статус NN_DETECT.
# Доп. аргументы передаются в auto_flight_nndt.py (напр. --master tcp:127.0.0.1:5770).
#
#   ./launch_2_auto_flight.sh

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${HERE}/auto_flight_nndt.py" "$@"
