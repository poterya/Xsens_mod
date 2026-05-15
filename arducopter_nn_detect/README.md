# ArduCopter `NN_DETECT` flight mode (mode 29 / `NNDT`)

Snapshot of the custom flight mode that drives the propeller-fault
detection pipeline of this repository inside ArduCopter (ArduPilot).

This directory is the deliverable: the mode itself does **not** live in
this Python project — it is patched into an upstream ArduPilot tree.
Everything needed to reproduce the build on a clean machine is included
here.

## What the mode does

* Holds horizontal position (Loiter-style) and altitude (Alt-Hold style)
  with hard-locked target — **all pilot stick input is ignored**, so the
  airframe stays put for clean accelerometer-based vibration analysis.
* State machine `WaitingForHover -> Detecting`:
  detector engages **only** when the copter is armed, in the air and the
  EKF velocity estimate stays below `|vxy| <= 0.30 m/s` and
  `|vz| <= 0.30 m/s` continuously for 3 seconds. Disarm / landing
  returns to waiting and flushes the buffer.
* In `Detecting` state a rolling window of 50 body-frame accelerometer
  samples is fed into a de-meaned RMS vibration metric, which is then
  forwarded to `vibration_fault()` — the integration hook for the
  sklearn-trained RandomForest / DecisionTree model. The MVP commits a
  simple RMS threshold so the pipeline is testable end-to-end.
* MAVLink `STATUSTEXT` reports once a second:
  * `NN_DETECT: engaged, take off and hover to start detection`
  * `NN_DETECT: waiting hover vxy=… vz=…`
  * `NN_DETECT: hover stable, detection started`
  * `NN_DETECT: ok vib=…` / `NN_DETECT: BROKEN vib=…`
  * `NN_DETECT: propeller fault detected` (one-shot CRITICAL on first
    transition)

## Mode identity

| Field | Value |
|---|---|
| `Mode::Number` | `29` (gap between `TURTLE=28` and reserved `offboard=30`) |
| Long name | `NN Detect` |
| Short name (4 chars) | `NNDT` |
| Compile flag | `MODE_NN_DETECT_ENABLED` (default `1`) |
| Requires position | yes |
| Manual throttle | no |
| Allows arming from this mode | no |
| `is_autopilot()` | yes |

## Repo layout of this snapshot

```
arducopter_nn_detect/
├── README.md                          this file
├── INSTALL.md                         step-by-step build & SITL test
├── patches/
│   └── 0001-add-nn-detect-mode.patch  unified patch (apply with `git am`)
└── files/
    └── ArduCopter/
        ├── Copter.h                   reference copy of modified files
        ├── config.h
        ├── mode.cpp
        ├── mode.h
        └── mode_nn_detect.cpp         the new implementation
```

The patch is the source of truth. The flat copies under `files/` are
provided so that the changes can be inspected without applying anything.

## Upstream version tested

Generated from the local ArduPilot tree at:

* commit `ArduCopter V4.8.0-dev` (`1b34668cc0`)
* branch `master` (upstream), patch lives on a feature branch `nn_detect`

The patch is small and self-contained (5 files, ~335 insertions), so it
applies cleanly on any recent ArduPilot master that still has:

* `class Mode` in `ArduCopter/mode.h`
* `Copter::mode_from_mode_num()` in `ArduCopter/mode.cpp`
* the `MODE_*_ENABLED` macro convention in `ArduCopter/config.h`
* `loiter_nav`, `pos_control`, `attitude_control` accessible from `Mode`

If upstream API has shifted (e.g. renamed `pos_control->D_*` methods),
fix-ups will be required — see `INSTALL.md` for guidance.

## Quick test in SITL

In one terminal:

```bash
mkdir -p /tmp/sitl_nndt && cd /tmp/sitl_nndt
~/ardupilot/build/sitl/bin/arducopter --model + --speedup 1 --slave 0 \
    --defaults ~/ardupilot/Tools/autotest/default_params/copter.parm \
    --sim-address=127.0.0.1 -I0
```

In another:

```bash
mavproxy.py --master tcp:127.0.0.1:5760 --console --map
```

Once `EKF3 IMU0 is using GPS` appears, in the MAVProxy prompt:

```
mode GUIDED
arm throttle
takeoff 5
# wait for ~5 m altitude in the HUD
mode NNDT      # equivalent: mode 29
```

You should see the STATUSTEXT sequence listed above, ending with
periodic `NN_DETECT: ok vib=…` lines.

## Integration roadmap

1. Replace `vibration_fault()` body with auto-generated C++ inference of
   the sklearn model exported from `train_detector.py` /
   `train_rf_detector.py`. Target header:
   `ArduCopter/nn_detect_model.h`.
2. Synchronise the feature definition between the C++ side and the
   Python side: the Python `realtime_detector` currently uses a
   `window_size=20` RMS; the C++ side uses `NN_WINDOW=50`. Pick one.
3. (optional) Send `NAMED_VALUE_FLOAT` telemetry (`vib`, `nn_fault`) in
   addition to `STATUSTEXT`, so a GCS can plot the metric live.
