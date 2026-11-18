#ifndef DENISE_FWI_MISFIT_HISTORY_H
#define DENISE_FWI_MISFIT_HISTORY_H

#include <math.h>

/* Only the current iteration and the two preceding indices can be used.
   Zero iteration tags mark unwritten slots after each frequency-stage reset. */
enum { FWI_MISFIT_SLOTS = 3 };

struct fwi_misfit_history {
    double value[FWI_MISFIT_SLOTS];
    int iteration[FWI_MISFIT_SLOTS];
};

static inline void fwi_misfit_record(struct fwi_misfit_history *history,
                                     int iter, double value)
{
    int slot = iter % FWI_MISFIT_SLOTS;
    history->value[slot] = value;
    history->iteration[slot] = iter;
}

static inline int fwi_misfit_should_stop(const struct fwi_misfit_history *history,
                                        int iter, int min_iter, int search_failed,
                                        float threshold, float *relative_change)
{
    int current, previous;
    double old, now;

    if (iter <= min_iter)
        return 0;
    if (search_failed)
        return 1;
    if (iter < 3)
        return 0;

    current = iter % FWI_MISFIT_SLOTS;
    previous = (iter - 2) % FWI_MISFIT_SLOTS;
    if (history->iteration[current] != iter ||
        history->iteration[previous] != iter - 2)
        return 0;

    old = history->value[previous];
    now = history->value[current];
    /* LNORM=5 can produce signed objectives. */
    if (!isfinite(old) || old == 0.0 || !isfinite(now))
        return 0;

    *relative_change = (float)fabs((old - now) / old);
    return *relative_change <= threshold;
}

#endif
