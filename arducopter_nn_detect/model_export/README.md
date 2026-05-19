# Model export pipeline

This directory contains the tooling that turns the trained PyTorch MLP
into C source code suitable for compilation inside ArduCopter.

The high-level design and the rationale for tool choice are in
[`../MODEL_INTEGRATION.md`](../MODEL_INTEGRATION.md). This file is the
operational recipe.

## Contents

| File | Purpose |
|---|---|
| `propeller_fault_mlp_keras_set05.pt` | Trained PyTorch weights (state_dict). |
| `feature_extraction.py` | Python reference for the 80-D feature vector (vendored from branch `nir`). |
| `export_via_onnx2c.py` | **Canonical exporter**: PyTorch → ONNX → onnx2c → `nn_detect_model_onnx2c.c`. |
| `export_mlp_to_cpp.py` | Legacy hand-written exporter; now used only as an independent reference for verifying the C++ feature extractor against the Python source of truth. |
| `cpp_test_harness.cpp` | Standalone driver: reads a 50-sample window from stdin, prints 80 features + `P(fault)`. |
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

```bash
cd model_export
python3 export_via_onnx2c.py
# Writes:
#   model_export/mlp.onnx                          (intermediate, gitignored)
#   files/ArduCopter/nn_detect_model_onnx2c.c      (~530 KB, committed)
```

The C file plus the thin C++ wrapper in
`files/ArduCopter/nn_detect_model.cpp` together implement the
`NNDetectModel::predict_proba(float[80]) → float` API consumed by the
flight mode. Nothing else needs to change.

## Verifying

```bash
python3 verify_against_cpp.py                              # synthetic
python3 verify_against_cpp.py --csv ../CSV_for_tests/deformed_1.csv
python3 verify_against_cpp.py --csv ../CSV_for_tests/normal1.csv
```

Each invocation:

1. Compiles `nn_detect_features.cpp` (C++), `nn_detect_model.cpp` (C++),
   `nn_detect_model_onnx2c.c` (C, by `gcc`), and `cpp_test_harness.cpp`
   (C++) into a single binary - the same files that ArduCopter's `waf`
   build picks up, with the same per-extension language rules.
2. Generates a few 50-sample windows (from CSV or synthetic).
3. Pushes each window through PyTorch and through the compiled binary,
   reporting `|P_py − P_cpp|` per window.

Tolerance: `|ΔP| ≤ 1e-4`. Typical: `~1e-7`.

## Updating the hand-written feature extractor

If `src/common/feature_extraction.py` on branch `nir` changes, the C++
mirror in `nn_detect_features.cpp` has to be re-synchronised. Run

```bash
python3 export_mlp_to_cpp.py
```

which prints the max diff between the Python source-of-truth and a
pure-Python re-implementation of the C++ extractor. If this diff goes
above `~1e-14` the extractor needs hand-editing in
`files/ArduCopter/nn_detect_features.cpp` to track the Python changes.

To regenerate the *reference* hand-written model + extractor (as a diff
target against the onnx2c output, never used by ArduCopter):

```bash
python3 export_mlp_to_cpp.py --write-legacy
# -> writes model_export/legacy_handwritten/{nn_detect_model.{h,cpp},
#                                            nn_detect_features.{h,cpp}}
```
