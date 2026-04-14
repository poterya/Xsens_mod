#!/usr/bin/env bash
# Установка зависимостей проекта Detection / Xsens_mod
#
# Использование:
#   chmod +x install-deps.sh
#   ./install-deps.sh
#
# С виртуальным окружением (рекомендуется):
#   cd "$(dirname "$0")"
#   python3 -m venv .venv
#   source .venv/bin/activate
#   ./install-deps.sh
#
# Поддерживайте список пакетов в актуальном состоянии при новых import из PyPI.
# Код: Xsens_mod/main.py, DataPacketParser.py, SerialHandler.py, csv_plot_browser.py

set -euo pipefail
cd "$(dirname "$0")"

python3 -m pip install -U pip
python3 -m pip install \
  "numpy>=1.26" \
  "matplotlib>=3.8" \
  "pandas>=2.0" \
  "pyserial>=3.5"
