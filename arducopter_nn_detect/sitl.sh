#!/bin/bash

if [ -z "$1" ]; then
    echo "Ошибка: не указано имя файла"
    echo "Использование: $0 <имя_файла>"
    exit 1
fi

FILENAME="$1"
# Путь должен быть к реальной папке в репозитории (не корень файловой системы /CSV_for_tests/…)
SNAPSHOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${FILENAME}" in
  *.csv) CSV_PATH="${SNAPSHOT}/CSV_for_tests/${FILENAME}" ;;
  *)      CSV_PATH="${SNAPSHOT}/CSV_for_tests/${FILENAME}.csv" ;;
esac

if [[ ! -f "${CSV_PATH}" ]]; then
  echo "Ошибка: нет файла ${CSV_PATH}" >&2
  echo "Подсказка: положите CSV в ${SNAPSHOT}/CSV_for_tests/ или укажите имя вида normal2 или normal2.csv" >&2
  exit 1
fi

export NN_DETECT_CSV="${CSV_PATH}"
export NN_DETECT_CSV_HOVER=1
export NN_DETECT_CSV_LOOP=1


~/ardupilot/build/sitl/bin/arducopter --model=quad --speedup=4 \
  --defaults=$HOME/ardupilot/Tools/autotest/default_params/copter.parm -I0