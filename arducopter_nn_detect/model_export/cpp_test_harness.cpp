// Standalone harness: read a window of (total, rms_x, rms_y, rms_z) samples
// from stdin (50 rows, space-separated), then print the resulting features
// and the MLP probability so the Python verification script can compare
// against the PyTorch reference. Linked against the same nn_detect_*.cpp
// files that ArduCopter compiles.

#include <cstdio>

#include "../files/ArduCopter/nn_detect_features.h"
#include "../files/ArduCopter/nn_detect_model.h"

int main()
{
    constexpr int N = NNDetectFeatures::WINDOW_SIZE;
    float total[N], x[N], y[N], z[N];
    for (int i = 0; i < N; i++) {
        if (std::scanf("%f %f %f %f", &total[i], &x[i], &y[i], &z[i]) != 4) {
            std::fprintf(stderr, "cpp_test_harness: short input at row %d\n", i);
            return 1;
        }
    }

    float feats[NNDetectFeatures::FEATURE_COUNT];
    NNDetectFeatures::extract(total, x, y, z, feats);

    std::printf("FEATURES");
    for (int i = 0; i < (int)NNDetectFeatures::FEATURE_COUNT; i++) {
        std::printf(" %.9e", (double)feats[i]);
    }
    std::printf("\n");

    const float p = NNDetectModel::predict_proba(feats);
    std::printf("PROBA %.9e\n", (double)p);
    return 0;
}
