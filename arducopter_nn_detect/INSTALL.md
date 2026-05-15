# Installing `NN_DETECT` on a clean ArduPilot tree

This snapshot adds a single feature branch worth of changes on top of
ArduPilot/ArduCopter. Two installation paths are provided; pick one.

The result is identical: a SITL `arducopter` binary that exposes mode
number 29 (`NNDT`).

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

You also need `mavproxy.py` and `pymavlink` for the smoke test:

```bash
pip3 install --user MAVProxy pymavlink
```

---

## Path A — apply the patch (recommended)

The patch lives in
[`patches/0001-add-nn-detect-mode.patch`](patches/0001-add-nn-detect-mode.patch).

```bash
cd ~/ardupilot

# create an isolated feature branch from upstream master
git fetch origin
git checkout -b nn_detect origin/master

# apply the patch (carries the original commit message and authorship)
git am /path/to/Xsens_mod/arducopter_nn_detect/patches/0001-add-nn-detect-mode.patch
```

If `git am` complains about whitespace or a slightly drifted upstream,
fall back to `git apply --3way`:

```bash
git apply --3way /path/to/.../0001-add-nn-detect-mode.patch
git add ArduCopter/
git commit -m "ArduCopter: add NN_DETECT (mode 29) flight mode"
```

Verify the branch state:

```bash
git log --oneline -1                  # should show the NN_DETECT commit
git status                            # clean
ls ArduCopter/mode_nn_detect.cpp      # exists
```

Build SITL:

```bash
./waf configure --board sitl
./waf copter
```

A clean rebuild takes ~30 s on a modern laptop; the patched binary lands
at `build/sitl/bin/arducopter`.

---

## Path B — drop in the file copies

If you prefer to integrate manually (e.g. cherry-pick by hand into a
fork that has already diverged from upstream master), the
[`files/ArduCopter/`](files/ArduCopter/) directory contains the final
state of every touched file. Copy them on top of your tree:

```bash
cp arducopter_nn_detect/files/ArduCopter/Copter.h           ~/ardupilot/ArduCopter/Copter.h
cp arducopter_nn_detect/files/ArduCopter/config.h           ~/ardupilot/ArduCopter/config.h
cp arducopter_nn_detect/files/ArduCopter/mode.cpp           ~/ardupilot/ArduCopter/mode.cpp
cp arducopter_nn_detect/files/ArduCopter/mode.h             ~/ardupilot/ArduCopter/mode.h
cp arducopter_nn_detect/files/ArduCopter/mode_nn_detect.cpp ~/ardupilot/ArduCopter/mode_nn_detect.cpp
```

Then build as in Path A.

This path overwrites the whole files instead of merging diffs, so it
will conflict with any unrelated local edits you have made to
`Copter.h`, `config.h`, `mode.cpp` or `mode.h`. Path A is safer.

---

## Smoke test in SITL

Terminal 1 — start `arducopter`:

```bash
mkdir -p /tmp/sitl_nndt && cd /tmp/sitl_nndt
~/ardupilot/build/sitl/bin/arducopter --model + --speedup 1 --slave 0 \
    --defaults ~/ardupilot/Tools/autotest/default_params/copter.parm \
    --sim-address=127.0.0.1 -I0
```

Terminal 2 — connect MAVProxy:

```bash
mavproxy.py --master tcp:127.0.0.1:5760 --console --map
```

Wait until the MAVProxy console shows `EKF3 IMU0 is using GPS` and
`Flight battery 100 percent`. Then in the MAVProxy prompt:

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
NN_DETECT: waiting hover vxy=0.04 vz=0.00
NN_DETECT: hover stable, detection started
NN_DETECT: ok vib=0.001
NN_DETECT: ok vib=0.001
...
```

Exit the mode with `mode LAND` or `mode RTL`.

### Automated test

A scripted MAVLink test that runs the full sequence and asserts every
expected STATUSTEXT line is checked-in next to this file as
`tools/test_nndt_flight.py` (TODO). Until then the manual sequence above
is enough to validate the patch.

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

* **In NNDT the copter stays at "waiting hover" forever.**
  EKF velocity stays above the hover threshold. Common causes: wind in
  the SITL parameters, an aggressive `WPNAV_*` setting from a previous
  test, or the copter not actually airborne. Land with `mode LAND`, fix
  the underlying issue and retry.

* **In NNDT the copter slowly drifts down (or refuses to hold alt).**
  Make sure you switched from `GUIDED` after takeoff, *not* from
  `STABILIZE` — in `STABILIZE` throttle is manual, so on entry to NNDT
  there is no climb-rate handover. Always go `GUIDED -> takeoff -> NNDT`.

---

## Uninstall

```bash
cd ~/ardupilot
git checkout master         # leaves the nn_detect branch intact
# or:
git branch -D nn_detect     # destroy the feature branch entirely
./waf copter                # rebuild stock binary
```
