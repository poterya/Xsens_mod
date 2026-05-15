# Запуск NN_DETECT с собственным CSV в SITL

Короткий рецепт: как скормить любой `vibration_log.csv` нейронке внутри
симулированного полёта ArduCopter SITL.

## Что нужно

- Собранный SITL с патчем `NN_DETECT`
  (см. [`INSTALL.md`](INSTALL.md) — пути «A» или «B»).
- Файл с вибрациями в формате
  `time_seconds,total_vibration,rms_x,rms_y,rms_z`
  (именно так пишет `methods/main.py` → `VibrationAnalyzer.save_log`).
- `pymavlink` (для теста-скрипта).

```bash
pip3 install --user pymavlink
```

## Два режима запуска

| Режим | Когда использовать | Что задавать |
|------|-------------------|--------------|
| **Bench** (стенд) | быстро прогнать CSV, без полёта | только `NN_DETECT_CSV` |
| **In-flight** (в полёте) | как на реальном борту: arm → takeoff → hover → NNDT | `NN_DETECT_CSV` + `NN_DETECT_CSV_HOVER=1` |

Опционально к любому из них: `NN_DETECT_CSV_LOOP=1` — зациклить файл.

---

## Вариант A — быстрый стенд (без полёта)

### Терминал 1. Запустить SITL с CSV

```bash
pkill -9 -f arducopter; sleep 1
mkdir -p /tmp/sitl_nndt && cd /tmp/sitl_nndt

NN_DETECT_CSV=/полный/путь/к/vibration_log.csv \
~/ardupilot/build/sitl/bin/arducopter \
    --model=quad --speedup=4 \
    --defaults=$HOME/ardupilot/Tools/autotest/default_params/copter.parm \
    -I0
```

Ждёшь, когда SITL напишет `Waiting for connection ....`

### Терминал 2. Переключить в NNDT

Сохрани в `/tmp/quick_nndt.py`:

```python
from pymavlink import mavutil
m = mavutil.mavlink_connection('tcp:127.0.0.1:5760')
m.wait_heartbeat()
m.mav.command_long_send(
    m.target_system, m.target_component,
    mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
    1, 29, 0, 0, 0, 0, 0)            # custom_mode = 29 (NNDT)
while True:
    msg = m.recv_match(type='STATUSTEXT', blocking=True, timeout=2.0)
    if msg and 'NN_DETECT' in msg.text:
        print(msg.text)
```

Запусти:

```bash
python3 /tmp/quick_nndt.py
```

Ожидаемые сообщения (для нормального CSV — `p ~= 0.02`, `ok`; для
сломанного — `p = 1.00`, `BROKEN`):

```
NN_DETECT: CSV replay /полный/путь/...
NN_DETECT: engaged, replaying vibration CSV
NN_DETECT: ok p=0.02 vx=0.2 vy=0.4 vz=0.5 clip=0
NN_DETECT: ok p=0.04 vx=0.2 vy=0.4 vz=0.4 clip=0
...
NN_DETECT: CSV exhausted, holding last sample
```

> В этом режиме арминг и взлёт **не нужны** — CSV проигрывается сразу
> после `mode NNDT`.

---

## Вариант B — настоящий симулированный полёт + CSV

Это сценарий, идентичный реальному борту: дрон взлетает в `GUIDED`,
зависает, переключается в `NNDT`, ждёт 3 с стабильного hover и только
тогда начинает «есть» CSV.

**Требование к сборке:** в дереве ArduPilot должен быть применён
актуальный `patches/nn_detect.patch` из этого снапшота — в коде есть
переменная окружения `NN_DETECT_CSV_HOVER`.

### Где лежат тестовые логи

Каталог в репозитории (полный путь на твоём ПК будет таким же
относительно клонированного дерева):

```text
Xsens_mod/arducopter_nn_detect/CSV_for_tests/
```

В нём уже лежат, например: `normal_1.csv`, `normal_2.csv`, …,
`deformed_1.csv`, … — указывай **только имя файла**, скрипт сам
подставит путь к `CSV_for_tests/`.

От тебя требуется одно действие перед запуском: перейти в каталог со
скриптами или вызывать их по абсолютному пути. Ниже — минимальный
рабочий сценарий.

### Терминал 1. SITL (один аргумент — имя файла CSV)

Быстрый вариант — обёртка (по умолчанию `normal_1.csv`):

```bash
cd /путь/к/Xsens_mod/arducopter_nn_detect/scripts
chmod +x launch_1_sitl_csv_hover.sh launch_2_auto_flight.sh   # один раз
./launch_1_sitl_csv_hover.sh              # normal_1.csv
./launch_1_sitl_csv_hover.sh deformed_1.csv
```

Эквивалентно прямому вызову:

```bash
./start_sitl_csv_hover.sh normal_1.csv
```

Скрипт сам выставит `NN_DETECT_CSV`, `NN_DETECT_CSV_HOVER=1` и по
умолчанию `NN_DETECT_CSV_LOOP=1`; рабочая директория SITL —
`/tmp/sitl_nndt_hover`.

Опционально:

| Переменная | Значение |
|------------|-----------|
| `ARDUPILOT` | путь к дереву ArduPilot (если не `~/ardupilot`) |
| `NN_DETECT_CSV_LOOP` | `0` — один проход по файлу, без зацикливания |
| `SITL_INSTANCE` | `1` второй симулятор (MAVLink `tcp:127.0.0.1:5770`) |
| `SPEEDUP` | коэффициент ускорения (по умолчанию `4`) |

Если файл лежит **не** в `CSV_for_tests/`, передай абсолютный путь:
`./start_sitl_csv_hover.sh /home/you/logs/my_log.csv`.

Жди строку вида `Waiting for connection ....` на порту SERIAL0.

### Терминал 2. Полёт (полностью автоматический)

```bash
cd /путь/к/Xsens_mod/arducopter_nn_detect/scripts
./launch_2_auto_flight.sh
```

Эквивалент: `python3 auto_flight_nndt.py` (все флаги можно дописать в конец
`launch_2_auto_flight.sh …`).

По умолчанию клиент слушает `tcp:127.0.0.1:5760` (соответствует
`-I0`). Для `SITL_INSTANCE=1` запускай так:

```bash
./launch_2_auto_flight.sh --master tcp:127.0.0.1:5770
```

Параметры: `--takeoff-alt 5`, `--listen 120` — сколько секунд после
переключения в NNDT печатать сообщения «NN_DETECT: …».
`--gps-timeout 180` — если долго висит на ожидании GPS.
`--alt-timeout 120` — если таймаут набора высоты.
`--no-relax-prearm` — не выставлять `ARMING_CHECK=0` в SITL.

##### Если второй терминал завис на «Ждём GPS fix»

- В первом терминале сообщение **`Waiting for internal clock bits to be set`**
  — это **не ошибка синхронизации времени**. Его выводит симулятор светодиода
  LP5562 (I2C) в SITL; на полёт и GPS оно напрямую не влияет.
- Обычно причина в том, что клиент очень долго не видит сообщение
  `GPS_RAW_INT` по TCP. Скрипт `auto_flight_nndt.py` после обновления сам
  шлёт `MAV_CMD_SET_MESSAGE_INTERVAL`, чтобы получать GPS (~5 Hz), и берёт
  компонент отправителя `MAV_COMP_ID_AUTOPILOT1`, если в heartbeat было `comp=0`.
  Обнови скрипт из репозитория и попробуй снова (`git pull`).
- Убедись, что на **TCP 5760 только один активный клиент** (нет второго
  `mavproxy` / второго экземпляра скрипта). Иначе SITL часто режет соединение
  после «Connection on serial port».

#### Способ 2b — MAVProxy вручную (без скрипта)

```bash
mavproxy.py --master tcp:127.0.0.1:5760 --console
```

После того как пройдут prearm-проверки (`EKF3 IMU0 is using GPS`):

```
mode GUIDED
arm throttle
takeoff 5
# подождать, пока высота HUD не будет ~5 м
mode NNDT
```

Ожидаемая лента сообщений:

```
NN_DETECT: CSV replay /полный/путь/... hover-gated
NN_DETECT: engaged, CSV replay armed — take off and hover
NN_DETECT: waiting hover vxy=0.04 vz=0.02
NN_DETECT: waiting hover vxy=0.03 vz=0.01
NN_DETECT: hover stable, detection started
NN_DETECT: filling window (15/50)
NN_DETECT: BROKEN p=1.00 vx=2.1 vy=2.4 vz=3.1 clip=0
NN_DETECT: BROKEN p=1.00 vx=2.7 vy=3.5 vz=7.7 clip=0
...
```

---

## Шпаргалка

| Что хочу | Команда |
|---------|---------|
| Прогнать CSV без полёта | `NN_DETECT_CSV=...` + `mode NNDT` |
| Полёт + CSV только по имени файла | **Вариант B**: `./launch_1_sitl_csv_hover.sh <имя>.csv` + `./launch_2_auto_flight.sh` |
| Прогнать CSV в полёте вручную | `NN_DETECT_CSV=... NN_DETECT_CSV_HOVER=1` + `arm` → `takeoff` → `mode NNDT` |
| Зациклить CSV | По умолчанию в скрипте вкл.; выкл.: `NN_DETECT_CSV_LOOP=0 ./start_sitl_csv_hover.sh …` |
| Освободить порт 5760 | `pkill -9 -f arducopter` |
| Запустить второй SITL | `SITL_INSTANCE=1 ./start_sitl_csv_hover.sh …` + `python3 auto_flight_nndt.py --master tcp:127.0.0.1:5770` |

Готовые тестовые CSV в этом репо лежат в каталоге
[`CSV_for_tests/`](CSV_for_tests/) (можно использовать только имя файла со
скриптом `start_sitl_csv_hover.sh`).

Примеры больших логов из ветки `methods` можно выгрузить так:

```bash
git show methods:Normal_mod/simulation_20260410_175920/vibration_log.csv   > /tmp/normal.csv
git show methods:Deformed_mod/front_left-old/simulation_20260409_155458/vibration_log.csv > /tmp/deformed.csv
```

Подставь любой свой `vibration_log.csv` — формат тот же, что пишет
`methods/main.py`.
