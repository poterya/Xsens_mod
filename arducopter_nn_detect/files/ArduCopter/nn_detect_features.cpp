// Implementation of the 73-feature extractor used by the NN_DETECT mode.
// Formulas are deliberately written to match numpy / pandas exactly so the
// RF inference in C++ produces the same probabilities as the offline Python
// training pipeline (methods/realtime_rf_detector.py + train_rf_detector.py).

#include "nn_detect_features.h"

#include <math.h>
#include <string.h>

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif

namespace NNDetectFeatures {

namespace {

constexpr uint16_t N = WINDOW_SIZE;     // 50
constexpr uint16_t FFT_BINS = N / 2 + 1; // 26
constexpr float    FSAMP = SAMPLE_RATE_HZ;

// Population mean.
inline float mean_of(const float *x)
{
    float s = 0.0f;
    for (uint16_t i = 0; i < N; i++) {
        s += x[i];
    }
    return s / static_cast<float>(N);
}

// Sort ascending into out[].
void sort_copy(const float *x, float out[N])
{
    memcpy(out, x, sizeof(float) * N);
    // simple insertion sort - N=50, dwarfed by FFT cost
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

// numpy.percentile linear interpolation on a pre-sorted array.
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
    float std_pop;     // numpy.std default ddof=0
    float vmin;
    float vmax;
    float range;
    float median;
    float q25;
    float q75;
    float iqr;
    float skew;        // pandas .skew()  (bias-corrected Fisher-Pearson)
    float kurtosis;    // pandas .kurtosis() (excess, bias-corrected)
};

// Compute mean/std/min/max + percentiles + skew + kurtosis.
void stats_block(const float *x, Stats &s)
{
    s.mean = mean_of(x);

    float m2 = 0.0f, m3 = 0.0f, m4 = 0.0f;
    float vmin = x[0], vmax = x[0];
    for (uint16_t i = 0; i < N; i++) {
        const float d = x[i] - s.mean;
        const float d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
        if (x[i] < vmin) vmin = x[i];
        if (x[i] > vmax) vmax = x[i];
    }
    m2 /= static_cast<float>(N);
    m3 /= static_cast<float>(N);
    m4 /= static_cast<float>(N);

    s.std_pop = sqrtf(m2);
    s.vmin = vmin;
    s.vmax = vmax;
    s.range = vmax - vmin;

    float sorted_buf[N];
    sort_copy(x, sorted_buf);
    s.median = percentile_sorted(sorted_buf, 50.0f);
    s.q25    = percentile_sorted(sorted_buf, 25.0f);
    s.q75    = percentile_sorted(sorted_buf, 75.0f);
    s.iqr    = s.q75 - s.q25;

    // pandas-style adjusted Fisher-Pearson skew (G1) and excess kurtosis (G2).
    // For N=50: skew_adj = sqrt(N*(N-1)) / (N-2); kurt_adj = (N-1)/((N-2)*(N-3))
    if (m2 > 0.0f) {
        const float n_f = static_cast<float>(N);
        const float g1 = m3 / powf(m2, 1.5f);
        const float g2 = (m4 / (m2 * m2)) - 3.0f;
        s.skew = sqrtf(n_f * (n_f - 1.0f)) / (n_f - 2.0f) * g1;
        s.kurtosis = ((n_f - 1.0f) / ((n_f - 2.0f) * (n_f - 3.0f))) *
                     ((n_f + 1.0f) * g2 + 6.0f);
    } else {
        // pandas returns NaN here, but for safety - and because the training
        // pipeline replaces NaN with 0 - we emit zeros.
        s.skew = 0.0f;
        s.kurtosis = 0.0f;
    }
}

struct Spectral {
    float dom_freq;
    float dom_mag;
    float spectral_energy;
    float spectral_entropy;
    float spectral_centroid;
};

// Real-input DFT magnitude of (x - mean(x)) at bins 0..FFT_BINS-1,
// matching np.fft.rfft. Then derive the 5 spectral features the model uses.
//
// The DFT inner sums are accumulated in double precision so that bin-
// boundary peak picking (np.argmax over magnitudes) matches numpy on
// hostile inputs like a pure sinusoid landing exactly between two bins.
void spectral_block(const float *x, Spectral &sp)
{
    double m = 0.0;
    for (uint16_t i = 0; i < N; i++) {
        m += static_cast<double>(x[i]);
    }
    m /= static_cast<double>(N);

    double mags[FFT_BINS];
    // O(N^2) DFT - N=50, FFT_BINS=26 ~ 1300 ops. Negligible vs RF inference.
    for (uint16_t k = 0; k < FFT_BINS; k++) {
        double re = 0.0, im = 0.0;
        const double w = -2.0 * M_PI *
                         static_cast<double>(k) / static_cast<double>(N);
        for (uint16_t n_ = 0; n_ < N; n_++) {
            const double xn = static_cast<double>(x[n_]) - m;
            re += xn * cos(w * static_cast<double>(n_));
            im += xn * sin(w * static_cast<double>(n_));
        }
        mags[k] = sqrt(re * re + im * im);
    }

    // power and totals
    double total_power = 0.0;
    double power[FFT_BINS];
    for (uint16_t k = 0; k < FFT_BINS; k++) {
        power[k] = mags[k] * mags[k];
        total_power += power[k];
    }
    const double tp_safe = total_power + 1e-12;

    // peak in bins [1, FFT_BINS-1] (DC excluded), match Python np.argmax(fft[1:])+1
    uint16_t peak_idx = 1;
    double   peak_mag = mags[1];
    for (uint16_t k = 2; k < FFT_BINS; k++) {
        if (mags[k] > peak_mag) {
            peak_mag = mags[k];
            peak_idx = k;
        }
    }

    // np.fft.rfftfreq(N, d=1/Fs) = k * Fs / N
    const double freq_step = static_cast<double>(FSAMP) / static_cast<double>(N);
    sp.dom_freq = static_cast<float>(static_cast<double>(peak_idx) * freq_step);
    sp.dom_mag  = static_cast<float>(peak_mag);
    sp.spectral_energy = static_cast<float>(total_power);

    double entropy = 0.0;
    double centroid = 0.0;
    const double inv_ln2 = 1.0 / log(2.0);
    for (uint16_t k = 0; k < FFT_BINS; k++) {
        const double p = power[k] / tp_safe;
        if (p > 0.0) {
            entropy -= p * (log(p) * inv_ln2);  // log2
        }
        centroid += (static_cast<double>(k) * freq_step) * power[k];
    }
    sp.spectral_entropy = static_cast<float>(entropy);
    sp.spectral_centroid = static_cast<float>(centroid / tp_safe);
}

inline float pearson_corr(const float *x, const float *y)
{
    const float mx = mean_of(x);
    const float my = mean_of(y);
    float sxy = 0.0f, sxx = 0.0f, syy = 0.0f;
    for (uint16_t i = 0; i < N; i++) {
        const float dx = x[i] - mx;
        const float dy = y[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    const float denom = sqrtf(sxx * syy);
    if (denom <= 0.0f) {
        return 0.0f;
    }
    return sxy / denom;
}

// Fill 16 features for a single channel starting at out[offset].
void fill_channel(const float *x, float *out)
{
    Stats s;
    Spectral sp;
    stats_block(x, s);
    spectral_block(x, sp);
    out[0]  = s.mean;
    out[1]  = s.std_pop;
    out[2]  = s.vmax;
    out[3]  = s.vmin;
    out[4]  = s.range;
    out[5]  = s.median;
    out[6]  = s.q25;
    out[7]  = s.q75;
    out[8]  = s.iqr;
    out[9]  = s.skew;
    out[10] = s.kurtosis;
    out[11] = sp.dom_freq;
    out[12] = sp.dom_mag;
    out[13] = sp.spectral_energy;
    out[14] = sp.spectral_entropy;
    out[15] = sp.spectral_centroid;
}

}  // namespace

void extract(const float total[WINDOW_SIZE],
             const float rms_x[WINDOW_SIZE],
             const float rms_y[WINDOW_SIZE],
             const float rms_z[WINDOW_SIZE],
             float out[FEATURE_COUNT])
{
    // 4 channels x 16 features each (indices 0..63 in feature_names order:
    // total_vibration, rms_x, rms_y, rms_z).
    fill_channel(total,  out + 0);
    fill_channel(rms_x,  out + 16);
    fill_channel(rms_y,  out + 32);
    fill_channel(rms_z,  out + 48);

    // Global features (indices 64..72).
    const float t_mean_safe = (out[0] > 1e-9f) ? out[0] : 1e-9f; // total_vibration_mean
    out[64] = out[16] / t_mean_safe;  // ratio_x_total = rms_x_mean / total_vibration_mean
    out[65] = out[32] / t_mean_safe;  // ratio_y_total
    out[66] = out[48] / t_mean_safe;  // ratio_z_total

    // total_vibration first differences: mean / max / std of |diff|.
    float d_mean = 0.0f, d_max = 0.0f, d_sumsq = 0.0f;
    {
        float diffs[WINDOW_SIZE - 1];
        for (uint16_t i = 0; i < N - 1; i++) {
            float d = total[i + 1] - total[i];
            if (d < 0.0f) {
                d = -d;
            }
            diffs[i] = d;
            d_mean += d;
            if (d > d_max) {
                d_max = d;
            }
        }
        d_mean /= static_cast<float>(N - 1);
        for (uint16_t i = 0; i < N - 1; i++) {
            const float dd = diffs[i] - d_mean;
            d_sumsq += dd * dd;
        }
    }
    out[67] = d_mean;
    out[68] = d_max;
    out[69] = sqrtf(d_sumsq / static_cast<float>(N - 1));  // numpy.std ddof=0

    out[70] = pearson_corr(rms_x, rms_y);
    out[71] = pearson_corr(rms_x, rms_z);
    out[72] = pearson_corr(rms_y, rms_z);

    // Final safety pass: NaN -> 0 (training pipeline does the same).
    for (uint16_t i = 0; i < FEATURE_COUNT; i++) {
        if (!isfinite(out[i])) {
            out[i] = 0.0f;
        }
    }
}

}  // namespace NNDetectFeatures
