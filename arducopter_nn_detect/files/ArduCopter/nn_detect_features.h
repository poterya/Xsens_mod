// Feature extractor for the NN_DETECT flight mode MLP classifier.
//
// Given four parallel rolling windows of WINDOW_SIZE samples each
// (total_vibration, rms_x, rms_y, rms_z), the extractor produces the
// FEATURE_COUNT-dimensional feature vector in the exact order returned by
// the Python reference implementation `src/common/feature_extraction.py`
// on branch `nir`.

#pragma once

#include <stdint.h>

namespace NNDetectFeatures {

constexpr uint16_t WINDOW_SIZE   = 50;
constexpr uint16_t FEATURE_COUNT = 80;
constexpr float    SAMPLE_RATE_HZ = 100.0f;

void extract(const float total[WINDOW_SIZE],
             const float rms_x[WINDOW_SIZE],
             const float rms_y[WINDOW_SIZE],
             const float rms_z[WINDOW_SIZE],
             float out[FEATURE_COUNT]);

}  // namespace NNDetectFeatures
