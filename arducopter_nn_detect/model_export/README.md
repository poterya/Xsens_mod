# Model export pipeline

This directory contains the tooling that turns the trained MLP detector
into C source code suitable for compilation inside ArduCopter.

The current model is `method_mlp.pkl` from branch `tests`: an
`sklearn.Pipeline` of `StandardScaler` + `MLPClassifier`
(46 inputs → 16 ReLU → 1 sigmoid), trained on 10-sample (0.1 s) windows.

## Contents

| File | Purpose |
|---|---|
| `method_mlp.pkl` | Trained sklearn Pipeline (StandardScaler + MLPClassifier), source of truth. |
| `propeller_fault_mlp_keras_set05.pt` | PyTorch mirror of the sklearn model, produced by `export_mlp_to_cpp.py`. Filename kept for pipeline compatibility. |
| `feature_extraction.py` | Python reference for the 46-D feature vector (copy of `features.py` from branch `tests`). |
| `export_mlp_to_cpp.py` | **Converter**: `method_mlp.pkl` → `.pt`. Mirrors the sklearn weights into a PyTorch `MLPDetector` (StandardScaler baked as `BatchNorm1d(46, affine=false)`). Verifies against sklearn on real CSV windows. |
| `export_via_onnx2c.py` | **Canonical exporter**: `.pt` → ONNX → onnx2c → `nn_detect_model_onnx2c.c`. |
| `cpp_test_harness.cpp` | Standalone driver: reads a 10-sample window from stdin, prints 46 features + `P(fault)`. |
| `verify_against_cpp.py` | Compiles the C/C++ files that ArduPilot will compile and checks bit-equivalence with PyTorch on real CSV windows. |

## Prerequisites

Python packages (user-local install is fine):

```bash
pip3 install --user torch onnx onnxruntime onnxscript numpy pandas
```

You also need `onnx2c` built locally. Below is the recipe we used on
Ubuntu 22.04 where `sudo apt install libprotobuf-dev` is unavailable
(headless / shared-dev box). On a normal machine just `sudo apt install
libprotobuf-dev protobuf-compiler` and skip steps 1-2.

### 0. Clone onnx2c

```bash
cd ~ && git clone --depth=1 https://github.com/kraiskil/onnx2c.git
cd onnx2c && git submodule update --init
```

### 1. Get protobuf headers without sudo (optional, only if you can't apt)

```bash
mkdir -p /tmp/protobuf_extract && cd /tmp/protobuf_extract
apt-get download libprotobuf-dev libprotobuf23 protobuf-compiler libprotoc23
for f in *.deb; do dpkg-deb -x "$f" ./root; done
# Headers   : /tmp/protobuf_extract/root/usr/include/google/protobuf/
# Libraries : /tmp/protobuf_extract/root/usr/lib/x86_64-linux-gnu/
# protoc    : /tmp/protobuf_extract/root/usr/bin/protoc
```

### 2. onnx2c CMakeLists patch (only against modern GCC)

Recent libstdc++ deprecates `std::is_pod`, which protobuf-3.12 headers
still use. onnx2c builds with `-Werror`, so this fails. Relax:

```bash
sed -i \
  -e 's/-Wall -Werror"/-Wall -Wno-deprecated-declarations"/' \
  -e 's/-Wall -Werror -Wfatal-errors"/-Wall -Wno-deprecated-declarations -Wfatal-errors"/' \
  ~/onnx2c/CMakeLists.txt
```

### 3. Build

```bash
export PROOT=/tmp/protobuf_extract/root/usr   # or /usr if you installed via apt
export LD_LIBRARY_PATH=$PROOT/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PATH=$PROOT/bin:$PATH

mkdir -p ~/onnx2c/build && cd ~/onnx2c/build
cmake .. \
  -DProtobuf_INCLUDE_DIR=$PROOT/include \
  -DProtobuf_LIBRARY=$PROOT/lib/x86_64-linux-gnu/libprotobuf.so \
  -DProtobuf_PROTOC_EXECUTABLE=$PROOT/bin/protoc \
  -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) onnx2c
```

This puts the `onnx2c` binary at `~/onnx2c/build/onnx2c`. The Python
exporter looks for it there by default; override via `ONNX2C_BIN` and
`PROTOBUF_LIB_DIR` environment variables if you installed it
elsewhere.

## Regenerating the model

Two steps, run from `model_export/`:

```bash
# 1) sklearn .pkl -> PyTorch .pt
python3 export_mlp_to_cpp.py
# Writes: propeller_fault_mlp_keras_set05.pt
#         (verifies PyTorch vs sklearn on real CSV windows)

# 2) .pt -> ONNX -> onnx2c -> C
python3 export_via_onnx2c.py
# Writes: model_export/mlp.onnx                       (intermediate, gitignored)
#         files/ArduCopter/nn_detect_model_onnx2c.c   (~29 KB, committed)
```

The C file plus the thin C++ wrapper in
`files/ArduCopter/nn_detect_model.cpp` together implement the
`NNDetectModel::predict_proba(float[46]) → float` API consumed by the
flight mode.

> **Degenerate features.** Four spectral features `spec_*_band_0_10`
> are the 0–10 Hz band on a demeaned 10-sample window — i.e. the DC bin,
> which is ~0 by construction (only float roundoff ~1e-32 remains). Their
> `StandardScaler` scale is ~1e-30, so naive normalisation overflows in
> float32. `export_mlp_to_cpp.py` therefore replaces those scales with
> 1.0 (so the features contribute ~0). This is the reason the PyTorch/C
> path differs from raw sklearn by ~4e-4 in probability — sklearn is
> overfitting amplified roundoff noise; the C path is the correct one.

## Verifying

```bash
python3 verify_against_cpp.py                              # real CSV windows
python3 verify_against_cpp.py --csv ../CSV_for_tests/deformed_1.csv
python3 verify_against_cpp.py --csv ../CSV_for_tests/normal1.csv
```

Each invocation:

1. Compiles `nn_detect_features.cpp` (C++), `nn_detect_model.cpp` (C++),
   `nn_detect_model_onnx2c.c` (C, by `gcc`), and `cpp_test_harness.cpp`
   (C++) into a single binary - the same files that ArduCopter's `waf`
   build picks up, with the same per-extension language rules.
2. Generates a few 10-sample windows (from CSV or synthetic).
3. Pushes each window through PyTorch and through the compiled binary,
   reporting `|P_py − P_cpp|` per window.

Tolerance: `|ΔP| ≤ 1e-4`. Typical: `~1e-10` (C ↔ PyTorch match exactly).
