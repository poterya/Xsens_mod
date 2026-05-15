#!/usr/bin/env bash
# Терминал 1: запуск SITL с hover-gated CSV replay (вариант B).
# По умолчанию — normal_1.csv из CSV_for_tests/
#
#   ./launch_1_sitl_csv_hover.sh
#   ./launch_1_sitl_csv_hover.sh deformed_2.csv

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/start_sitl_csv_hover.sh" "${1:-normal_1.csv}"
