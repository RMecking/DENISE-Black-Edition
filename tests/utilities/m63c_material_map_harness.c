#include "fd.h"

void err(char err_text[]) {
    fprintf(stderr, "%s\n", err_text);
    abort();
}

void m63c6a_init_mapping(struct q_tau_mapping *mapping, int mode,
                         int mechanisms, const float *frequencies,
                         float fmin, float fmax, float df) {
    init_q_tau_mapping(mapping, mode, mechanisms, frequencies, fmin, fmax, df);
}

float m63c6a_q_to_tau(float q, const struct q_tau_mapping *mapping) {
    return q_to_tau(q, mapping);
}

double m63c6a_q_derivative(float q, const struct q_tau_mapping *mapping) {
    return q_to_tau_derivative(q, mapping);
}
