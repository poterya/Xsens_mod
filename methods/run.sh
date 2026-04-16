#!/usr/bin/env bash
# Запуск скриптов детекции вибрации из корня репозитория.
#
# Использование:
#   chmod +x methods/run.sh
#   ./methods/run.sh train
#   ./methods/run.sh detect
#
# Если есть .venv в корне проекта — он подключается автоматически.
# Интерпретатор: переменная PYTHON (по умолчанию python3).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi

usage() {
  cat <<'EOF'
Команды:

  deps        — установить зависимости (install-deps.sh)
  train       — обучить дерево решений → propeller_fault_model.pkl
  train-rf    — обучить Random Forest → propeller_fault_rf.pkl
  detect      — онлайн-детекция (дерево, нужен propeller_fault_model.pkl)
  detect-rf   — онлайн-детекция (RF, нужен propeller_fault_rf.pkl)
  record      — запись вибрации с Xsens (main.py)

Аргументы после команды передаются в соответствующий Python-скрипт, например:
  ./methods/run.sh train --max-depth 8
  ./methods/run.sh train-rf --output my_model.pkl

Переменные окружения:
  PYTHON   — python3 по умолчанию
EOF
}

cmd="${1:-}"
if [[ -n "$cmd" ]]; then
  shift
fi

case "$cmd" in
  "" | -h | --help | help)
    usage
    exit 0
    ;;
  deps)
    exec bash "$ROOT/install-deps.sh"
    ;;
  train)
    exec "$PYTHON" "$ROOT/train_detector.py" "$@"
    ;;
  train-rf)
    exec "$PYTHON" "$ROOT/train_rf_detector.py" "$@"
    ;;
  detect)
    exec "$PYTHON" "$ROOT/realtime_detector.py" "$@"
    ;;
  detect-rf)
    exec "$PYTHON" "$ROOT/realtime_rf_detector.py" "$@"
    ;;
  record)
    exec "$PYTHON" "$ROOT/main.py" "$@"
    ;;
  *)
    echo "Неизвестная команда: $cmd" >&2
    echo >&2
    usage >&2
    exit 1
    ;;
esac
