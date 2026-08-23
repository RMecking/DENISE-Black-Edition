from __future__ import annotations

import struct

import pytest

from tests.utilities.sh_free_surface_runtime import (
    binary32,
    denise_ricker_reference,
    post_source_quarters,
)


def _bits(value: float) -> int:
    return struct.unpack("!I", struct.pack("!f", value))[0]


def test_binary32_reference_quantization_is_explicit_and_deterministic():
    assert _bits(binary32(1.0 + 2.0**-24)) == 0x3F800000
    assert _bits(binary32(1.0 + 3.0 * 2.0**-24)) == 0x3F800002
    assert _bits(binary32(2.0**-149)) == 0x00000001
    assert _bits(binary32(2.0**-150)) == 0x00000000
    assert _bits(binary32(-(2.0**-150))) == 0x80000000


def test_mandatory_ricker_uses_one_based_denise_timesteps_and_locked_cutoff():
    reference = denise_ricker_reference(
        nt=5201,
        dt_s=0.0005,
        frequency_hz=8.0,
        amplitude=1.0,
        timeshift_s=0.0,
        quellart=1,
        n_order=0,
    )
    assert reference.peak_timestep == 375
    assert reference.n_off == 1257
    assert reference.samples[1256] != 0.0
    assert reference.samples[1257] == 0.0
    assert reference.n_off * 0.0005 == pytest.approx(0.6285)


def test_post_source_quarters_are_exact_and_exhaustive():
    quarters = post_source_quarters(nt=5201, n_off=1257)
    assert quarters.quarter_size == 986
    assert quarters.inclusive_bounds == (
        (1258, 2243),
        (2244, 3229),
        (3230, 4215),
        (4216, 5201),
    )
    flattened = [
        sample
        for first, last in quarters.inclusive_bounds
        for sample in range(first, last + 1)
    ]
    assert flattened == list(range(1258, 5202))


def test_post_source_quarters_reject_invalid_lengths():
    with pytest.raises(ValueError, match="at least four"):
        post_source_quarters(nt=1260, n_off=1257)
    with pytest.raises(ValueError, match="divisible by four"):
        post_source_quarters(nt=5200, n_off=1257)


@pytest.mark.parametrize(
    ("quellart", "n_order", "message"),
    ((2, 0, "QUELLART=1"), (1, 1, "N_ORDER=0")),
)
def test_source_off_scope_rejects_unsupported_source_semantics(
    quellart, n_order, message
):
    with pytest.raises(ValueError, match=message):
        denise_ricker_reference(
            nt=5201,
            dt_s=0.0005,
            frequency_hz=8.0,
            quellart=quellart,
            n_order=n_order,
        )
