// Implementation of the 80-feature extractor used by the NN_DETECT mode MLP.
// Formulas are deliberately written to match numpy / scipy.stats exactly so
// the C++ inference produces the same logits as the Python reference
// (src/common/feature_extraction.py on branch `nir`).

#include "nn_detect_features.h"

#include <math.h>
#include <string.h>

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif

namespace NNDetectFeatures {

namespace {

constexpr uint16_t N = WINDOW_SIZE;       // 50
constexpr uint16_t FFT_BINS = N / 2 + 1;  // 26
constexpr float    FSAMP = SAMPLE_RATE_HZ;

inline float mean_of(const float *x)
{
    float s = 0.0f;
    for (uint16_t i = 0; i < N; i++) {
        s += x[i];
    }
    return s / static_cast<float>(N);
}

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
    float vmax;
    float vmin;
    float range;
    float median;
    float iqr;         // q75 - q25
    float p90;
    float skew;        // scipy.stats.skew(bias=False)
    float kurtosis;    // scipy.stats.kurtosis(fisher=True, bias=False)
};

void stats_block(const float *x, Stats &s)
{
    // Accumulate in double so a 50-sample constant window produces an exact
    // zero variance, matching numpy's behaviour (which would otherwise diverge
    // from the C++ extractor on near-constant signals).
    double sum = 0.0;
    double vmin = x[0], vmax = x[0];
    for (uint16_t i = 0; i < N; i++) {
        sum += static_cast<double>(x[i]);
        if (x[i] < vmin) vmin = x[i];
        if (x[i] > vmax) vmax = x[i];
    }
    const double mean_d = sum / static_cast<double>(N);

    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (uint16_t i = 0; i < N; i++) {
        const double d = static_cast<double>(x[i]) - mean_d;
        const double d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= static_cast<double>(N);
    m3 /= static_cast<double>(N);
    m4 /= static_cast<double>(N);

    s.mean = static_cast<float>(mean_d);
    s.std_pop = static_cast<float>(sqrt(m2));
    s.vmax = static_cast<float>(vmax);
    s.vmin = static_cast<float>(vmin);
    s.range = static_cast<float>(vmax - vmin);

    float sorted_buf[N];
    sort_copy(x, sorted_buf);
    s.median = percentile_sorted(sorted_buf, 50.0f);
    const float q25 = percentile_sorted(sorted_buf, 25.0f);
    const float q75 = percentile_sorted(sorted_buf, 75.0f);
    s.iqr = q75 - q25;
    s.p90 = percentile_sorted(sorted_buf, 90.0f);

    // scipy.stats skew/kurtosis with bias=False match pandas-style formulas
    // already used in the previous (RF) snapshot of this extractor.
    // Python guards: if size >= 8 and std(values) > 1e-12 - compute, else 0.
    // For N=50 the size check is always satisfied.
    if (sqrt(m2) > 1e-12) {
        const double n_d = static_cast<double>(N);
        const double g1 = m3 / pow(m2, 1.5);
        const double g2 = (m4 / (m2 * m2)) - 3.0;
        const double skew_adj = sqrt(n_d * (n_d - 1.0)) / (n_d - 2.0) * g1;
        const double kurt_adj = ((n_d - 1.0) / ((n_d - 2.0) * (n_d - 3.0))) *
                                ((n_d + 1.0) * g2 + 6.0);
        s.skew = static_cast<float>(skew_adj);
        s.kurtosis = static_cast<float>(kurt_adj);
    } else {
        s.skew = 0.0f;
        s.kurtosis = 0.0f;
    }
}

struct Spectral {
    float dom_freq;
    float e_0_5;
    float e_5_10;
    float e_10_20;
    float e_20_50;
    float spec_entropy;
};

// Real-input DFT magnitude of (x - mean(x)) at bins 0..FFT_BINS-1, matching
// np.fft.rfft. Then derive the 6 spectral features the MLP uses.
//
// Inner sums accumulate in double precision so that the band-energy ratios
// and entropy match the Python reference bit-for-bit on every reasonable
// input.
void spectral_block(const float *x, Spectral &sp)
{
    // Python: if std(values - mean) < 1e-12 - emit zeros.
    double m = 0.0;
    for (uint16_t i = 0; i < N; i++) {
        m += static_cast<double>(x[i]);
    }
    m /= static_cast<double>(N);

    double var_acc = 0.0;
    for (uint16_t i = 0; i < N; i++) {
        const double d = static_cast<double>(x[i]) - m;
        var_acc += d * d;
    }
    const double std_after_demean = sqrt(var_acc / static_cast<double>(N));

    sp.dom_freq = 0.0f;
    sp.e_0_5 = 0.0f;
    sp.e_5_10 = 0.0f;
    sp.e_10_20 = 0.0f;
    sp.e_20_50 = 0.0f;
    sp.spec_entropy = 0.0f;
    if (!(std_after_demean > 1e-12)) {
        return;
    }

    double mags[FFT_BINS];
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

    double power[FFT_BINS];
    double total_power = 0.0;
    for (uint16_t k = 0; k < FFT_BINS; k++) {
        power[k] = mags[k] * mags[k];
        total_power += power[k];
    }
    const double tp_safe = total_power + 1e-12;

    // np.argmax(spectrum[1:]) + 1 over linear magnitudes (not power).
    uint16_t peak_idx = 1;
    double peak_mag = mags[1];
    for (uint16_t k = 2; k < FFT_BINS; k++) {
        if (mags[k] > peak_mag) {
            peak_mag = mags[k];
            peak_idx = k;
        }
    }
    const double freq_step = static_cast<double>(FSAMP) / static_cast<double>(N);
    sp.dom_freq = static_cast<float>(static_cast<double>(peak_idx) * freq_step);

    // Energy in bands [lo, hi) as fractions of the (unsafe-normalised) total.
    auto band_frac = [&](double lo, double hi) -> double {
        double s = 0.0;
        for (uint16_t k = 0; k < FFT_BINS; k++) {
            const double f = static_cast<double>(k) * freq_step;
            if (f >= lo && f < hi) {
                s += power[k];
            }
        }
        return s / tp_safe;
    };
    sp.e_0_5   = static_cast<float>(band_frac(0.0, 5.0));
    sp.e_5_10  = static_cast<float>(band_frac(5.0, 10.0));
    sp.e_10_20 = static_cast<float>(band_frac(10.0, 20.0));
    sp.e_20_50 = static_cast<float>(band_frac(20.0, 50.0));

    // Spectral entropy uses natural log and the same tp_safe denominator,
    // mirroring the Python reference exactly. Bins with p <= 1e-12 are
    // dropped (they would contribute 0 in the limit anyway).
    double entropy = 0.0;
    for (uint16_t k = 0; k < FFT_BINS; k++) {
        const double p = power[k] / tp_safe;
        if (p > 1e-12) {
            entropy -= p * log(p);
        }
    }
    sp.spec_entropy = static_cast<float>(entropy);
}

inline float pearson_corr(const float *x, const float *y)
{
    // Double-precision accumulators so a constant window with float roundoff
    // doesn't get a spurious non-zero correlation (Python's _safe_corr returns
    // 0 when either operand has std < 1e-9; in double precision a strictly
    // constant signal still produces exact zero std here).
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
    if (std_x < 1e-9 || std_y < 1e-9) {
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

    // 0..9  : total_vibration_*
    out[0] = st_total.mean;
    out[1] = st_total.std_pop;
    out[2] = st_total.vmax;
    out[3] = st_total.vmin;
    out[4] = st_total.range;
    out[5] = st_total.median;
    out[6] = st_total.iqr;
    out[7] = st_total.p90;
    out[8] = st_total.skew;
    out[9] = st_total.kurtosis;

    // 10..19 : rms_x_*
    out[10] = st_x.mean;
    out[11] = st_x.std_pop;
    out[12] = st_x.vmax;
    out[13] = st_x.vmin;
    out[14] = st_x.range;
    out[15] = st_x.median;
    out[16] = st_x.iqr;
    out[17] = st_x.p90;
    out[18] = st_x.skew;
    out[19] = st_x.kurtosis;

    // 20..29 : rms_y_*
    out[20] = st_y.mean;
    out[21] = st_y.std_pop;
    out[22] = st_y.vmax;
    out[23] = st_y.vmin;
    out[24] = st_y.range;
    out[25] = st_y.median;
    out[26] = st_y.iqr;
    out[27] = st_y.p90;
    out[28] = st_y.skew;
    out[29] = st_y.kurtosis;

    // 30..39 : rms_z_*
    out[30] = st_z.mean;
    out[31] = st_z.std_pop;
    out[32] = st_z.vmax;
    out[33] = st_z.vmin;
    out[34] = st_z.range;
    out[35] = st_z.median;
    out[36] = st_z.iqr;
    out[37] = st_z.p90;
    out[38] = st_z.skew;
    out[39] = st_z.kurtosis;

    // 40..46 : ratios / axis_dominance. Python clamps each divisor to 1e-9.
    const float t_mean_safe = (st_total.mean > 1e-9f) ? st_total.mean : 1e-9f;
    const float y_safe      = (st_y.mean     > 1e-9f) ? st_y.mean     : 1e-9f;
    const float z_safe      = (st_z.mean     > 1e-9f) ? st_z.mean     : 1e-9f;
    out[40] = st_x.mean / t_mean_safe;        // ratio_x_total
    out[41] = st_y.mean / t_mean_safe;        // ratio_y_total
    out[42] = st_z.mean / t_mean_safe;        // ratio_z_total
    out[43] = st_x.mean / y_safe;             // ratio_x_y
    out[44] = st_x.mean / z_safe;             // ratio_x_z
    out[45] = st_y.mean / z_safe;             // ratio_y_z
    float axis_max = st_x.mean;
    if (st_y.mean > axis_max) axis_max = st_y.mean;
    if (st_z.mean > axis_max) axis_max = st_z.mean;
    out[46] = axis_max / t_mean_safe;         // axis_dominance

    // 47..49 : total_vibration first-difference stats (mean/max/std of |diff|).
    // Python uses numpy.std (ddof=0) over the absolute differences.
    float d_mean = 0.0f, d_max = 0.0f, d_sumsq = 0.0f;
    {
        float diffs[N - 1];
        for (uint16_t i = 0; i < N - 1; i++) {
            float d = total[i + 1] - total[i];
            if (d < 0.0f) d = -d;
            diffs[i] = d;
            d_mean += d;
            if (d > d_max) d_max = d;
        }
        d_mean /= static_cast<float>(N - 1);
        for (uint16_t i = 0; i < N - 1; i++) {
            const float dd = diffs[i] - d_mean;
            d_sumsq += dd * dd;
        }
    }
    out[47] = d_mean;
    out[48] = d_max;
    out[49] = sqrtf(d_sumsq / static_cast<float>(N - 1));

    // 50..55 : total_vibration spectral features.
    Spectral sp_total, sp_x, sp_y, sp_z;
    spectral_block(total, sp_total);
    spectral_block(rms_x, sp_x);
    spectral_block(rms_y, sp_y);
    spectral_block(rms_z, sp_z);

    out[50] = sp_total.dom_freq;
    out[51] = sp_total.e_0_5;
    out[52] = sp_total.e_5_10;
    out[53] = sp_total.e_10_20;
    out[54] = sp_total.e_20_50;
    out[55] = sp_total.spec_entropy;

    // 56..61 : rms_x spectral features.
    out[56] = sp_x.dom_freq;
    out[57] = sp_x.e_0_5;
    out[58] = sp_x.e_5_10;
    out[59] = sp_x.e_10_20;
    out[60] = sp_x.e_20_50;
    out[61] = sp_x.spec_entropy;

    // 62..67 : rms_y spectral features.
    out[62] = sp_y.dom_freq;
    out[63] = sp_y.e_0_5;
    out[64] = sp_y.e_5_10;
    out[65] = sp_y.e_10_20;
    out[66] = sp_y.e_20_50;
    out[67] = sp_y.spec_entropy;

    // 68..73 : rms_z spectral features.
    out[68] = sp_z.dom_freq;
    out[69] = sp_z.e_0_5;
    out[70] = sp_z.e_5_10;
    out[71] = sp_z.e_10_20;
    out[72] = sp_z.e_20_50;
    out[73] = sp_z.spec_entropy;

    // 74..76 : Pearson correlations between axes.
    out[74] = pearson_corr(rms_x, rms_y);
    out[75] = pearson_corr(rms_x, rms_z);
    out[76] = pearson_corr(rms_y, rms_z);

    // 77..79 : axis asymmetries normalised by total_vibration_mean.
    out[77] = (st_x.mean - st_y.mean) / t_mean_safe;
    out[78] = (st_x.mean - st_z.mean) / t_mean_safe;
    out[79] = (st_y.mean - st_z.mean) / t_mean_safe;

    // Final safety pass: NaN/Inf -> 0 (training pipeline does the same).
    for (uint16_t i = 0; i < FEATURE_COUNT; i++) {
        if (!isfinite(out[i])) {
            out[i] = 0.0f;
        }
    }
}

}  // namespace NNDetectFeatures
