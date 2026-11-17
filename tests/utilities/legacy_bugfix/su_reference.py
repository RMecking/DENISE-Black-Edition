"""Independent native SU record fixture recipe for SU-WRITER-BOUNDARY-01."""

from math import atan2, floor, pi
import struct


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def round_nearest(value):
    return floor(value + 0.5)


def fixture(case):
    ntr, available, stride, count = (case[key] for key in ("traces", "available", "stride", "samples"))
    kind = case["producer"]
    dt = float32(float32(.001) * stride)
    micros = round_nearest(dt * 1e6)
    raw, expected = bytearray(), bytearray()

    for row in range(ntr):
        words = [0x3e000000 | ((131*i + 913*row) & 0x7fffff) |
                 (0x80000000 if (i+row) % 2 else 0) for i in range(available)]
        words[0] = words[stride-1] = 0x3fa51a0a + row
        if count >= 4:
            words[2*stride-1] = 0x80000000
            words[3*stride-1] = 0x00000011 + row
        raw += struct.pack(f"<{available}I", *words)
        header = bytearray(240)
        struct.pack_into("<h", header, 28, 1)
        struct.pack_into("<HH", header, 114, count, micros)
        struct.pack_into("<f", header, 180, dt if kind == "SOURCE" else micros / 1e6)
        if kind == "SOURCE":
            struct.pack_into("<i", header, 0, 1)
        else:
            if kind == "LOCAL":
                tracl = cdp = ntr - row
                pos = tracl - 1
            elif kind == "GLOBAL":
                tracl, pos = 21 + row, row
                cdp = 0 if ntr == 1 else 300 + round_nearest(pos / 2)
            else:
                tracl, pos, cdp = 1, 0, 0
            x, y = 800 + 4*pos, 800 + 3*pos
            azimuth = round_nearest(1000 * (180 / pi) * atan2(4*pos, 3*pos))
            for offset, value in [(0, tracl), (20, cdp), (36, 5000*pos), (40, 1000*y),
                                  (48, 800000), (60, azimuth), (72, 800000), (76, 800000),
                                  (80, 1000*x), (84, 1000*y)]:
                struct.pack_into("<i", header, offset, value)
            struct.pack_into("<hh", header, 68, -1000, -1000)
        expected += header
        expected += struct.pack(f"<{count}I", *words[stride-1:count*stride:stride])
    return bytes(raw), bytes(expected)
