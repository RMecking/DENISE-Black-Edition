#ifndef DENISE_SU_WRITE_H
#define DENISE_SU_WRITE_H

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

enum { SU_WRITE_HEADER_BYTES = 240 };
typedef char su_write_native_layout[
    offsetof(segy, data) == SU_WRITE_HEADER_BYTES && sizeof(float) == 4 ? 1 : -1];

struct su_sampling {
    int count;
    int stride;
    unsigned short interval_us;
};

static void su_output_error(FILE *stream, char message[])
{
    if (stream)
        fclose(stream);
    err(message);
}

static struct su_sampling su_output_sampling(FILE *stream, int available, int stride,
                                             float dt, int use_time, float time)
{
    struct su_sampling sampling;
    int steps = available;
    double interval;

    if (!stream)
        su_output_error(stream, "Could not open SU output");
    if (available < 1 || stride < 1)
        su_output_error(stream, "Invalid SU output sample count or stride");
    if (!isfinite(dt) || dt <= 0.0f)
        su_output_error(stream, "Invalid SU output time interval");

    if (use_time) {
        /* Preserve float TIME/DT and the original iround convention. */
        double rounded = floor((double)(time / dt) + 0.5);
        if (!isfinite(rounded) || rounded < 1.0 || rounded > INT_MAX)
            su_output_error(stream, "Invalid SU output time extent");
        steps = (int)rounded;
    }
    sampling.count = steps / stride;
    if (sampling.count < 1 || sampling.count > SU_NFLTS ||
        sampling.count > available / stride)
        su_output_error(stream, "SU output sample count outside supported range");

    interval = floor(((float)stride * dt) * 1.0e6 + 0.5);
    if (!isfinite(interval) || interval > UINT16_MAX)
        su_output_error(stream, "SU output interval does not fit the header");
    sampling.interval_us = (unsigned short)interval;
    sampling.stride = stride;
    return sampling;
}

static void su_write_trace(FILE *stream, segy *trace, const float *samples,
                           struct su_sampling sampling)
{
    int j;
    size_t count = (size_t)sampling.count;

    /* All callers supply a one-based row. Count and stride were checked
       before geometry or samples were accessed. */
    if (sampling.stride == 1) {
        memcpy(trace->data, samples + 1, count * sizeof(trace->data[0]));
    } else {
        for (j = 0; j < sampling.count; ++j)
            memcpy(&trace->data[j], &samples[(j + 1) * sampling.stride],
                   sizeof(trace->data[0]));
    }
    if (fwrite(trace, 1, SU_WRITE_HEADER_BYTES, stream) != SU_WRITE_HEADER_BYTES ||
        ferror(stream))
        su_output_error(stream, "Incomplete SU output header");
    if (fwrite(trace->data, sizeof(trace->data[0]), count, stream) != count ||
        ferror(stream))
        su_output_error(stream, "Incomplete SU output samples");
}

static void su_close_output(FILE *stream)
{
    if (fclose(stream) != 0)
        err("Could not close SU output");
}

#endif
