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
* In `Detecting` state, the per-axis vibration RMS from
  `AP_InertialSensor::get_vibration_levels()` (identical to the MAVLink
  `VIBRATION` message and the `VIBE` log) is pushed into a 50-deep
  ring buffer. Once the ring is full, the 73-D feature vector is
  extracted (`nn_detect_features.cpp`) and fed to the exported
  RandomForest model (`nn_detect_model.{h,cpp}` — 200 trees,
  `max_depth=12`). The class-1 probability is EMA-smoothed and
  compared against `NN_DEFORMED_THRESHOLD = 0.35`.
* MAVLink `STATUSTEXT` reports once a second:
  * `NN_DETECT: engaged, …`
  * `NN_DETECT: waiting hover vxy=… vz=…`
  * `NN_DETECT: hover stable, detection started`
  * `NN_DETECT: filling window (k/50)`
  * `NN_DETECT: ok p=… vx=… vy=… vz=… clip=…`
  * `NN_DETECT: BROKEN p=… vx=… vy=… vz=… clip=…`
  * `NN_DETECT: propeller fault detected` (one-shot CRITICAL on first
    transition)

## SITL CSV replay

For end-to-end testing without a real vibration source, set the
`NN_DETECT_CSV` environment variable before launching `arducopter`
(SITL builds only). The file must be in the format produced by
`methods/main.py VibrationAnalyzer.save_log`:

```
time_seconds,total_vibration,rms_x,rms_y,rms_z
15.925,0.831173,0.325665,0.571104,0.508558
...
```

When CSV replay is active the detector skips the hover-stability gate
and starts inferring immediately after `mode NNDT`. Use
`NN_DETECT_CSV_LOOP=1` to loop the file. The IMU branch is unchanged;
on real flight controllers the file is never opened.

Verified end-to-end with samples from the `methods` branch:

| CSV | Result |
|-----|--------|
| `Normal_mod/.../vibration_log.csv` | mostly `ok`, `p ~= 0.02` |
| `Deformed_mod/.../vibration_log.csv` | `BROKEN`, `p = 1.00` |

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
│   └── nn_detect.patch                unified patch (two commits, apply with `git am`)
└── files/
    └── ArduCopter/
        ├── Copter.h                   reference copy of modified files
        ├── config.h
        ├── mode.cpp
        ├── mode.h
        ├── mode_nn_detect.cpp         the new implementation
        ├── nn_detect_features.h       73-feature extractor (declarations)
        ├── nn_detect_features.cpp     73-feature extractor (impl)
        ├── nn_detect_model.h          exported RF model (declarations)
        └── nn_detect_model.cpp        exported RF model data (~2.9 MB)
```

The patch is the source of truth. The flat copies under `files/` are
provided so that the changes can be inspected without applying anything.

## Upstream version tested

Generated from the local ArduPilot tree at:

* upstream base `ArduCopter V4.8.0-dev` (`1b34668cc0`)
* feature branch `nn_detect` (two commits — the mode skeleton + the RF
  inference / CSV replay integration)

The patch applies cleanly on any recent ArduPilot master that still has:

* `class Mode` in `ArduCopter/mode.h`
* `Copter::mode_from_mode_num()` in `ArduCopter/mode.cpp`
* the `MODE_*_ENABLED` macro convention in `ArduCopter/config.h`
* `loiter_nav`, `pos_control`, `attitude_control` accessible from `Mode`
* `AP_InertialSensor::get_vibration_levels()` /
  `get_accel_clip_count()` /
  `get_first_usable_accel()` accessors

If upstream API has shifted (e.g. renamed `pos_control->D_*` methods),
fix-ups will be required — see `INSTALL.md` for guidance.

## Quick test in SITL

See [`INSTALL.md`](INSTALL.md) for the full procedure (both the live
IMU path and the CSV replay path).
