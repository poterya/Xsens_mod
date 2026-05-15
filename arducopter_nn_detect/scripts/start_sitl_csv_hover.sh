#!/usr/bin/env bash
# Запуск SITL ArduCopter с режимом NN_DETECT: CSV replay + hover-gate
# (см. RUN_CSV.md, вариант B).
#
# Использование (из любой директории):
#   ./start_sitl_csv_hover.sh normal_1.csv
#
# По умолчанию файл ищется в ../CSV_for_tests/ рядом с этим скриптом.
# Если передать абсолютный путь или путь вида ./foo.csv и файл есть — будет использован он.
#
# Переменные окружения (опционально):
#   ARDUPILOT    — дерево ArduPilot (по умолчанию ~/ardupilot)
#   SITL_INSTANCE — экземпляр SITL: 0 -> порт 5760, 1 -> 5770, ...
#   NN_DETECT_CSV_LOOP — 1 (по умолчанию) зациклить CSV, 0 один проход
#   SPEEDUP — коэффициент ускорения симуляции (по умолчанию 4)

set -euo pipefail

NAME="${1:-}"
if [[ -z "${NAME}" ]]; then
  echo "usage: $(basename "$0") <csv_filename_or_path>" >&2
  echo "" >&2
  echo "Пример: $(basename "$0") normal_1.csv" >&2
  echo "CSV по умолчанию: CSV_for_tests/ внутри каталога arducopter_nn_detect" >&2
  exit 2
fi

SNAPSHOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV_DIR="${SNAPSHOT_ROOT}/CSV_for_tests"

if [[ -f "${NAME}" ]]; then
  CSV="$(realpath "${NAME}")"
elif [[ -f "${CSV_DIR}/${NAME}" ]]; then
  CSV="$(realpath "${CSV_DIR}/${NAME}")"
else
  echo "error: файл не найден ни как ${CSV_DIR}/${NAME}, ни как ${NAME}" >&2
  echo "CSV_for_tests содержит:" >&2
  ls -1 "${CSV_DIR}" 2>/dev/null || true
  exit 1
fi

ARDUPILOT="${ARDUPILOT:-${HOME}/ardupilot}"
BIN="${ARDUPILOT}/build/sitl/bin/arducopter"
PARM="${ARDUPILOT}/Tools/autotest/default_params/copter.parm"

if [[ ! -x "${BIN}" ]]; then
  echo "error: нет исполняемого ${BIN}" >&2
  echo "Собери SITL: cd \"${ARDUPILOT}\" && ./waf configure --board sitl && ./waf copter" >&2
  exit 1
fi
if [[ ! -f "${PARM}" ]]; then
  echo "error: не найден ${PARM}" >&2
  exit 1
fi

INSTANCE="${SITL_INSTANCE:-0}"
SPEEDUP="${SPEEDUP:-4}"
LOOP="${NN_DETECT_CSV_LOOP:-1}"

WORKDIR="${SITL_WORKDIR:-/tmp/sitl_nndt_hover}"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

echo "CSV: ${CSV}"
echo "arducopter: ${BIN}"
echo "instance: ${INSTANCE} (SERIAL0 TCP $((5760 + INSTANCE * 10)))"
echo "NN_DETECT_CSV_HOVER=1 NN_DETECT_CSV_LOOP=${LOOP}"
echo ""

export NN_DETECT_CSV="${CSV}"
export NN_DETECT_CSV_HOVER=1
export NN_DETECT_CSV_LOOP="${LOOP}"

echo "Остановка старых SITL (если есть)..."
pkill -9 -f '[b]in/arducopter' 2>/dev/null || true
sleep 1

exec "${BIN}" --model=quad --speedup="${SPEEDUP}" \
  --defaults="${PARM}" \
  "-I${INSTANCE}"
