// Implementation of the 46-feature extractor used by the NN_DETECT mode MLP.
// Formulas are deliberately written to match numpy / scipy.stats exactly so
// the C++ inference produces the same logits as the Python reference
// (features.extract_features on branch `tests`).
//
// Feature order (matches feature_names in method_mlp.pkl):
//   0..5    total_{mean,std,max,min,p95,range}
//   6..11   rms_x_{mean,std,max,min,p95,range}
//   12..17  rms_y_{...}
//   18..23  rms_z_{...}
//   24..27  spec_total_band_{0_10,10_20,20_35,35_50}
//   28..31  spec_x_band_{...}
//   32..35  spec_y_band_{...}
//   36..39  spec_z_band_{...}
//   40..42  corr_{x_y, x_z, y_z}
//   43..45  corr_total_{x, y, z}

#include "nn_detect_features.h"

#include <math.h>
#include <string.h>

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif

namespace NNDetectFeatures {

namespace {

constexpr uint16_t N = WINDOW_SIZE;       // 10
constexpr uint16_t FFT_BINS = N / 2 + 1;  // 6
constexpr float    FSAMP = SAMPLE_RATE_HZ;

void sort_copy(const float *x, float out[N])
{
    memcpy(out, x, sizeof(float) * N);
    for (uint16_t i = 1; i < N; i++) {
        const float v = out[i];
        int16_t j = static_cast<int16_t>(i) - 1;
        while (j >= 0 && out[j] > v) {
            out[j + 1] = out[j];
            j--;
        }
        out[j + 1] = v;
    }
}

float percentile_sorted(const float *sorted_x, float q_pct)
{
    const float pos = (q_pct / 100.0f) * static_cast<float>(N - 1);
    const int16_t lo = static_cast<int16_t>(pos);
    const int16_t hi = (lo + 1 < static_cast<int16_t>(N)) ? (lo + 1) : lo;
    const float frac = pos - static_cast<float>(lo);
    return sorted_x[lo] + (sorted_x[hi] - sorted_x[lo]) * frac;
}

struct Stats {
    float mean;
    float std_pop;
    float vmax;
    float vmin;
    float p95;
    float range;
};

void stats_block(const float *x, Stats &s)
{
    double sum = 0.0;
    double vmin = x[0], vmax = x[0];
    for (uint16_t i = 0; i < N; i++) {
        sum += static_cast<double>(x[i]);
        if (x[i] < vmin) vmin = x[i];
        if (x[i] > vmax) vmax = x[i];
    }
    const double mean_d = sum / static_cast<double>(N);

    double m2 = 0.0;
    for (uint16_t i = 0; i < N; i++) {
        const double d = static_cast<double>(x[i]) - mean_d;
        m2 += d * d;
    }
    m2 /= static_cast<double>(N);

    s.mean = static_cast<float>(mean_d);
    s.std_pop = static_cast<float>(sqrt(m2));
    s.vmax = static_cast<float>(vmax);
    s.vmin = static_cast<float>(vmin);
    s.range = static_cast<float>(vmax - vmin);

    float sorted_buf[N];
    sort_copy(x, sorted_buf);
    s.p95 = percentile_sorted(sorted_buf, 95.0f);
}

// power[k] = |rfft(x - mean(x))[k]|^2 / N, bins k = 0..FFT_BINS-1.
// Band feature = sum_{k: freq[k] in [lo, hi)} power[k]   (NOT normalised).
struct SpecBands {
    float b0_10;
    float b10_20;
    float b20_35;
    float b35_50;
};

void spec_bands(const float *x, SpecBands &sb)
{
    double m = 0.0;
    for (uint16_t i = 0; i < N; i++) {
        m += static_cast<double>(x[i]);
    }
    m /= static_cast<double>(N);

    double power[FFT_BINS];
    for (uint16_t k = 0; k < FFT_BINS; k++) {
        double re = 0.0, im = 0.0;
        const double w = -2.0 * M_PI *
                         static_cast<double>(k) / static_cast<double>(N);
        for (uint16_t n_ = 0; n_ < N; n_++) {
            const double xn = static_cast<double>(x[n_]) - m;
            re += xn * cos(w * static_cast<double>(n_));
            im += xn * sin(w * static_cast<double>(n_));
        }
        power[k] = (re * re + im * im) / static_cast<double>(N);
    }

    const double freq_step = static_cast<double>(FSAMP) / static_cast<double>(N);
    auto band_sum = [&](double lo, double hi) -> double {
        double s = 0.0;
        for (uint16_t k = 0; k < FFT_BINS; k++) {
            const double f = static_cast<double>(k) * freq_step;
            if (f >= lo && f < hi) {
                s += power[k];
            }
        }
        return s;
    };
    sb.b0_10  = static_cast<float>(band_sum(0.0, 10.0));
    sb.b10_20 = static_cast<float>(band_sum(10.0, 20.0));
    sb.b20_35 = static_cast<float>(band_sum(20.0, 35.0));
    sb.b35_50 = static_cast<float>(band_sum(35.0, 50.0));
}

inline float pearson_corr(const float *x, const float *y)
{
    double sumx = 0.0, sumy = 0.0;
    for (uint16_t i = 0; i < N; i++) {
        sumx += static_cast<double>(x[i]);
        sumy += static_cast<double>(y[i]);
    }
    const double mx = sumx / static_cast<double>(N);
    const double my = sumy / static_cast<double>(N);
    double sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (uint16_t i = 0; i < N; i++) {
        const double dx = static_cast<double>(x[i]) - mx;
        const double dy = static_cast<double>(y[i]) - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    const double n_d = static_cast<double>(N);
    const double std_x = sqrt(sxx / n_d);
    const double std_y = sqrt(syy / n_d);
    if (std_x < 1e-12 || std_y < 1e-12) {
        return 0.0f;
    }
    return static_cast<float>((sxy / n_d) / (std_x * std_y));
}

}  // namespace

void extract(const float total[WINDOW_SIZE],
             const float rms_x[WINDOW_SIZE],
             const float rms_y[WINDOW_SIZE],
             const float rms_z[WINDOW_SIZE],
             float out[FEATURE_COUNT])
{
    Stats st_total, st_x, st_y, st_z;
    stats_block(total, st_total);
    stats_block(rms_x, st_x);
    stats_block(rms_y, st_y);
    stats_block(rms_z, st_z);

    out[0]  = st_total.mean;
    out[1]  = st_total.std_pop;
    out[2]  = st_total.vmax;
    out[3]  = st_total.vmin;
    out[4]  = st_total.p95;
    out[5]  = st_total.range;

    out[6]  = st_x.mean;
    out[7]  = st_x.std_pop;
    out[8]  = st_x.vmax;
    out[9]  = st_x.vmin;
    out[10] = st_x.p95;
    out[11] = st_x.range;

    out[12] = st_y.mean;
    out[13] = st_y.std_pop;
    out[14] = st_y.vmax;
    out[15] = st_y.vmin;
    out[16] = st_y.p95;
    out[17] = st_y.range;

    out[18] = st_z.mean;
    out[19] = st_z.std_pop;
    out[20] = st_z.vmax;
    out[21] = st_z.vmin;
    out[22] = st_z.p95;
    out[23] = st_z.range;

    SpecBands sp_total, sp_x, sp_y, sp_z;
    spec_bands(total, sp_total);
    spec_bands(rms_x, sp_x);
    spec_bands(rms_y, sp_y);
    spec_bands(rms_z, sp_z);

    out[24] = sp_total.b0_10;
    out[25] = sp_total.b10_20;
    out[26] = sp_total.b20_35;
    out[27] = sp_total.b35_50;

    out[28] = sp_x.b0_10;
    out[29] = sp_x.b10_20;
    out[30] = sp_x.b20_35;
    out[31] = sp_x.b35_50;

    out[32] = sp_y.b0_10;
    out[33] = sp_y.b10_20;
    out[34] = sp_y.b20_35;
    out[35] = sp_y.b35_50;

    out[36] = sp_z.b0_10;
    out[37] = sp_z.b10_20;
    out[38] = sp_z.b20_35;
    out[39] = sp_z.b35_50;

    out[40] = pearson_corr(rms_x, rms_y);
    out[41] = pearson_corr(rms_x, rms_z);
    out[42] = pearson_corr(rms_y, rms_z);
    out[43] = pearson_corr(total, rms_x);
    out[44] = pearson_corr(total, rms_y);
    out[45] = pearson_corr(total, rms_z);

    for (uint16_t i = 0; i < FEATURE_COUNT; i++) {
        if (!isfinite(out[i])) {
            out[i] = 0.0f;
        }
    }
}

}  // namespace NNDetectFeatures
