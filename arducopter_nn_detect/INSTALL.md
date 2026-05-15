# Installing `NN_DETECT` on a clean ArduPilot tree

This snapshot adds a single feature branch worth of changes on top of
ArduPilot/ArduCopter. Two installation paths are provided; pick one.

The result is identical: a SITL `arducopter` binary that exposes mode
number 29 (`NNDT`) with the exported RandomForest inference + 73-feature
extractor wired in, plus a SITL-only CSV vibration replay for testing.

---

## Prerequisites

A standard ArduPilot dev environment for Linux. If you do not have one
yet, follow the official guide once
(<https://ardupilot.org/dev/docs/building-setup-linux.html>):

```bash
git clone --recursive https://github.com/ArduPilot/ardupilot.git ~/ardupilot
cd ~/ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile
git submodule update --init --recursive
```

You also need `pymavlink` (and optionally `MAVProxy`) for the smoke
tests:

```bash
pip3 install --user MAVProxy pymavlink
```

---

## Path A — apply the patch (recommended)

The patch lives in
[`patches/nn_detect.patch`](patches/nn_detect.patch) and contains two
commits:

1. `ArduCopter: add NN_DETECT (mode 29) flight mode` — the mode
   skeleton + Loiter-style hover + hover-stability state machine.
2. `ArduCopter: NN_DETECT integrate RF inference + SITL CSV replay` —
   the exported RandomForest (200 trees, 73 features), the
   `nn_detect_features` / `nn_detect_model` modules, and the
   SITL CSV replay.

```bash
cd ~/ardupilot

# create an isolated feature branch from upstream master
git fetch origin
git checkout -b nn_detect origin/master

# apply the patch (carries the original commit messages and authorship)
git am /path/to/Xsens_mod/arducopter_nn_detect/patches/nn_detect.patch
```

If `git am` complains about whitespace or a slightly drifted upstream,
fall back to `git apply --3way`:

```bash
git apply --3way /path/to/.../nn_detect.patch
git add ArduCopter/
git commit -m "ArduCopter: NN_DETECT mode + RF inference + SITL CSV replay"
```

Verify the branch state:

```bash
git log --oneline -3                  # 2 NN_DETECT commits on top of master
git status                            # clean
ls ArduCopter/mode_nn_detect.cpp      # exists
ls ArduCopter/nn_detect_model.cpp     # exists, ~2.9 MB
```

Build SITL:

```bash
./waf configure --board sitl
./waf copter
```

A clean rebuild takes ~30 s on a modern laptop; the patched binary
lands at `build/sitl/bin/arducopter`.

---

## Path B — drop in the file copies

If you prefer to integrate manually (e.g. cherry-pick by hand into a
fork that has already diverged from upstream master), the
[`files/ArduCopter/`](files/ArduCopter/) directory contains the final
state of every touched/added file. Copy them on top of your tree:

```bash
DST=~/ardupilot/ArduCopter
SRC=arducopter_nn_detect/files/ArduCopter

cp $SRC/Copter.h               $DST/Copter.h
cp $SRC/config.h               $DST/config.h
cp $SRC/mode.cpp               $DST/mode.cpp
cp $SRC/mode.h                 $DST/mode.h
cp $SRC/mode_nn_detect.cpp     $DST/mode_nn_detect.cpp
cp $SRC/nn_detect_features.h   $DST/nn_detect_features.h
cp $SRC/nn_detect_features.cpp $DST/nn_detect_features.cpp
cp $SRC/nn_detect_model.h      $DST/nn_detect_model.h
cp $SRC/nn_detect_model.cpp    $DST/nn_detect_model.cpp
```

Then build as in Path A.

This path overwrites whole files instead of merging diffs, so it will
conflict with any unrelated local edits you have made to `Copter.h`,
`config.h`, `mode.cpp` or `mode.h`. Path A is safer.

---

## Smoke test 1 — live IMU path

Terminal 1 — start `arducopter`:

```bash
mkdir -p /tmp/sitl_nndt && cd /tmp/sitl_nndt
~/ardupilot/build/sitl/bin/arducopter --model=quad --speedup=1 \
    --defaults=$HOME/ardupilot/Tools/autotest/default_params/copter.parm \
    -I0
```

Terminal 2 — connect MAVProxy:

```bash
mavproxy.py --master tcp:127.0.0.1:5760 --console --map
```

Wait until the MAVProxy console shows `EKF3 IMU0 is using GPS`. Then in
the MAVProxy prompt:

```
mode GUIDED
arm throttle
takeoff 5
```

When the HUD shows altitude ~5 m and the copter has stopped climbing:

```
mode NNDT
```

(`mode 29` works too.) STATUSTEXT messages should appear in the
console:

```
NN_DETECT: engaged, take off and hover to start detection
NN_DETECT: waiting hover vxy=0.02 vz=0.00
NN_DETECT: waiting hover vxy=0.03 vz=0.00
NN_DETECT: hover stable, detection started
NN_DETECT: filling window (12/50)
NN_DETECT: ok p=0.04 vx=0.0 vy=0.0 vz=0.0 clip=0
...
```

Exit the mode with `mode LAND` or `mode RTL`.

> **Note.** SITL feeds the IMU with zero base vibration, which is an
> out-of-distribution input for the RF model trained on real flights.
> Use the CSV replay path below to validate detection on realistic
> data.

---

## Smoke test 2 — CSV replay (SITL only)

Set `NN_DETECT_CSV` to any CSV in the format produced by
`methods/main.py VibrationAnalyzer.save_log`:

```
time_seconds,total_vibration,rms_x,rms_y,rms_z
15.925,0.831173,0.325665,0.571104,0.508558
...
```

Then launch SITL with that env var set:

```bash
mkdir -p /tmp/sitl_nndt_csv && cd /tmp/sitl_nndt_csv

NN_DETECT_CSV=/path/to/vibration_log.csv \
~/ardupilot/build/sitl/bin/arducopter --model=quad --speedup=4 \
    --defaults=$HOME/ardupilot/Tools/autotest/default_params/copter.parm \
    -I0
```

Optional: `NN_DETECT_CSV_LOOP=1` to loop the file.

Drive the simulator with a tiny pymavlink helper — no arming, no
takeoff is required, the detector engages straight after `mode NNDT`:

```python
# replay_test.py
from pymavlink import mavutil
m = mavutil.mavlink_connection('tcp:127.0.0.1:5760')
m.wait_heartbeat()
m.mav.command_long_send(
    m.target_system, m.target_component,
    mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
    1, 29, 0, 0, 0, 0, 0)               # base_mode=CUSTOM_MODE_ENABLED, custom_mode=29
while True:
    msg = m.recv_match(type='STATUSTEXT', blocking=True, timeout=1.0)
    if msg and 'NN_DETECT' in msg.text:
        print(msg.text)
```

```bash
python3 replay_test.py
```

Expected output for a normal-prop log:

```
NN_DETECT: CSV replay /path/to/normal.csv
NN_DETECT: engaged, replaying vibration CSV
NN_DETECT: ok p=0.00 vx=0.2 vy=0.4 vz=0.5 clip=0
NN_DETECT: ok p=0.04 vx=0.2 vy=0.4 vz=0.4 clip=0
...
NN_DETECT: CSV exhausted, holding last sample
```

For a deformed-prop log every line should read `BROKEN p=1.00`:

```
NN_DETECT: BROKEN p=1.00 vx=1.6 vy=2.1 vz=3.9 clip=0
NN_DETECT: BROKEN p=1.00 vx=1.6 vy=2.1 vz=3.9 clip=0
...
```

---

## Troubleshooting

* **`git am` fails with `patch does not apply`.**
  Upstream `mode.h` has shifted. Open the rejected hunk, find the
  closest matching context in your tree and re-apply by hand, or use
  `git apply --3way` (see Path A).

* **Build error `pos_control->D_set_max_speed_accel_m not declared`.**
  Upstream renamed the position-controller API. `mode_nn_detect.cpp`
  mirrors `mode_loiter.cpp` 1:1; open your local
  `ArduCopter/mode_loiter.cpp` and copy the matching call signatures
  over.

* **SITL is stuck at `Waiting for internal clock bits to be set`.**
  A previous SITL run did not release TCP port 5760. Kill any stale
  `arducopter` / `mavproxy` processes (`pkill -9 -f arducopter`,
  `pkill -9 -f mavproxy.py`) or start the next run with another
  instance (`-I 1` -> port 5770).

* **`set_mode -> NN_DETECT` is rejected.**
  ArduCopter refuses mode changes until EKF + GPS are stable. Wait for
  the `EKF3 IMU0 is using GPS` console line, then retry.

* **In NNDT (live IMU path) the copter stays at "waiting hover"
  forever.** EKF velocity stays above the hover threshold. Common
  causes: wind in the SITL parameters, an aggressive `WPNAV_*` setting
  from a previous test, or the copter not actually airborne. Land with
  `mode LAND`, fix the underlying issue and retry. The CSV-replay path
  bypasses this gate entirely.

* **In NNDT the copter slowly drifts down (or refuses to hold alt).**
  Make sure you switched from `GUIDED` after takeoff, *not* from
  `STABILIZE` — in `STABILIZE` throttle is manual, so on entry to NNDT
  there is no climb-rate handover. Always go `GUIDED -> takeoff -> NNDT`.

* **CSV replay: the file opens but no detections appear.**
  Check the file format — the header `time_seconds,total_vibration,
  rms_x,rms_y,rms_z` must be present, and rows must be parseable as 5
  comma-separated floats (or 3 if you only have `rms_x,rms_y,rms_z`).
  Unparseable rows are skipped silently.

---

## Uninstall

```bash
cd ~/ardupilot
git checkout master         # leaves the nn_detect branch intact
# or:
git branch -D nn_detect     # destroy the feature branch entirely
./waf copter                # rebuild stock binary
```
