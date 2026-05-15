// Feature extractor for the NN_DETECT flight mode RandomForest classifier.
//
// Given four parallel rolling windows of WINDOW_SIZE samples each
// (total_vibration, rms_x, rms_y, rms_z), the extractor produces the
// 73-dimensional feature vector in the exact order expected by the model
// exported by methods/export_rf_to_cpp.py (see feature_names in the .pkl).
//
// The implementation mirrors numpy/pandas: population mean/std/var,
// linear-interpolation percentiles, pandas-style adjusted Fisher-Pearson
// skew and adjusted excess kurtosis, DFT of (signal - mean) of length
// WINDOW_SIZE sampled at SAMPLE_RATE_HZ.
#pragma once

#include <stdint.h>

namespace NNDetectFeatures {

constexpr uint16_t WINDOW_SIZE   = 50;
constexpr uint16_t FEATURE_COUNT = 73;
constexpr float    SAMPLE_RATE_HZ = 100.0f;

void extract(const float total[WINDOW_SIZE],
             const float rms_x[WINDOW_SIZE],
             const float rms_y[WINDOW_SIZE],
             const float rms_z[WINDOW_SIZE],
             float out[FEATURE_COUNT]);

}  // namespace NNDetectFeatures
