"""Numerical assertions carried over from the accepted SU reader regression."""

import math
import struct

N, CELLS = 48, 18 * 20
SCIENCE_RECORD = struct.Struct(f"<d{N}f{N}f{N}d{3 * CELLS}f")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_science(path):
    result = path.read_bytes()
    require(len(result) == 2 * SCIENCE_RECORD.size, "Wrong science evidence size")
    rows = list(SCIENCE_RECORD.iter_unpack(result))
    for case, row in enumerate(rows):
        objective = row[0]
        observed = row[1:1+N]
        synthetic = row[1+N:1+2*N]
        residual = row[1+2*N:1+3*N]
        expected_observed = [0.125 * case] + [0.0] * (N - 1)
        expected_observed[8], expected_observed[-1] = 1 / 64, -3 / 128
        require(struct.pack(f"<{N}f", *observed) ==
                struct.pack(f"<{N}f", *expected_observed), "Observation bits")
        require(all(math.isfinite(v) for v in row), "Nonfinite science result")
        expected = [0.0] + [
            struct.unpack("<f", struct.pack("<f", s - d))[0]
            for s, d in zip(synthetic[1:], expected_observed[1:])
        ]
        require(struct.pack(f"<{N}d", *residual) ==
                struct.pack(f"<{N}d", *expected), "Chronological residual bits")
        objective_ref = math.fsum(0.5 * v * v for v in expected)
        require(objective > 0 and abs(objective - objective_ref) <=
                64 * math.ulp(objective_ref), "Objective versus independent sum")
        require(any(v != 0 for v in synthetic), "No synthetic signal")
        require(any(v != 0 for v in row[1+3*N:]), "No material gradient")

    first, second = (result[i:i+SCIENCE_RECORD.size]
                     for i in range(0, len(result), SCIENCE_RECORD.size))
    require(first[:8] == second[:8], "Objective delta is not bitwise zero")
    require(first[12:8+4*N] == second[12:8+4*N], "Unexpected observation delta")
    require(first[8+4*N:] == second[8+4*N:],
            "Synthetic, active residual or material gradient changed")
    gradients = rows[0][1+3*N:]
    return {
        "sample1_observed_A_B": [rows[0][1], rows[1][1]],
        "sample1_raw_synthetic_minus_observed_delta": -0.125,
        "sample1_cotangent_A_B": [rows[0][1+2*N], rows[1][1+2*N]],
        "objective_A_B": [rows[0][0], rows[1][0]],
        "active_residual_delta_maxabs": 0,
        "gradient_delta_maxabs": [0, 0, 0],
        "gradient_maxabs_A": [
            max(map(abs, gradients[i:i+CELLS])) for i in range(0, len(gradients), CELLS)
        ],
    }

