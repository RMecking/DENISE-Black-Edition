/* Fixed-relaxation-frequency Q to GSLS relaxation-strength conversion. */

#include "fd.h"

void init_q_tau_mapping(struct q_tau_mapping *mapping, int mode, int mechanisms,
                        const float *relaxation_frequencies_hz,
                        float fmin_hz, float fmax_hz, float df_hz) {
    int index, mechanism, sample_count;
    double sum_a = 0.0, sum_ab = 0.0, sum_aa = 0.0;

    if (mapping == NULL) err(" Q parameterization mapping is NULL. ");
    mapping->mode = mode;
    mapping->sample_count = 0;
    mapping->inverse_tau_per_q = 0.0;
    mapping->inverse_tau_offset = 0.0;

    if (mode == Q_PARAMETERIZATION_LEGACY) return;
    if (mode != Q_PARAMETERIZATION_PHYSICAL)
        err(" Q_PARAMETERIZATION_MODE must be 0 (legacy) or 1 (physical-Q). ");
    if (mechanisms < 1 || relaxation_frequencies_hz == NULL)
        err(" Physical-Q parameterization requires at least one relaxation mechanism. ");
    if (!(fmin_hz > 0.0f && fmax_hz >= fmin_hz && df_hz > 0.0f))
        err(" Physical-Q parameterization requires 0 < Q_APPROX_FMIN <= Q_APPROX_FMAX and Q_APPROX_DF > 0. ");
    for (mechanism = 1; mechanism <= mechanisms; mechanism++)
        if (!(relaxation_frequencies_hz[mechanism] > 0.0f))
            err(" Physical-Q parameterization requires positive FL values. ");

    sample_count = (int)floor(((double)fmax_hz - fmin_hz) / df_hz + 1.0e-12) + 1;
    for (index = 0; index < sample_count; index++) {
        double frequency_hz = (double)fmin_hz + index * (double)df_hz;
        double omega = 2.0 * PI * frequency_hz;
        double A = 0.0, B = 0.0;
        for (mechanism = 1; mechanism <= mechanisms; mechanism++) {
            double theta = 1.0 / (2.0 * PI * relaxation_frequencies_hz[mechanism]);
            double omega_theta = omega * theta;
            double divisor = 1.0 + omega_theta * omega_theta;
            A += omega_theta * omega_theta / divisor;
            B += omega_theta / divisor;
        }
        if (!(B > 0.0) || !isfinite(B)) err(" Invalid physical-Q GSLS coefficient. ");
        {
            double a = 1.0 / B;
            double b = A / B;
            sum_a += a;
            sum_ab += a * b;
            sum_aa += a * a;
        }
    }
    if (!(sum_aa > 0.0) || !isfinite(sum_aa)) err(" Invalid physical-Q least-squares denominator. ");
    mapping->sample_count = sample_count;
    mapping->inverse_tau_per_q = sum_a / sum_aa;
    mapping->inverse_tau_offset = -sum_ab / sum_aa;
}

float q_to_tau(float target_q, const struct q_tau_mapping *mapping) {
    double inverse_tau;
    if (mapping == NULL) err(" Q parameterization mapping is NULL. ");
    if (!(target_q > 0.0f) || !isfinite(target_q)) err(" Qp/Qs model values must be finite and positive. ");
    if (mapping->mode == Q_PARAMETERIZATION_LEGACY) return (float)(2.0 / (double)target_q);
    if (mapping->mode != Q_PARAMETERIZATION_PHYSICAL) err(" Invalid Q parameterization mode. ");
    inverse_tau = mapping->inverse_tau_per_q * target_q + mapping->inverse_tau_offset;
    if (!(inverse_tau > 0.0) || !isfinite(inverse_tau))
        err(" Physical target Q does not yield a finite positive GSLS tau. ");
    return (float)(1.0 / inverse_tau);
}
