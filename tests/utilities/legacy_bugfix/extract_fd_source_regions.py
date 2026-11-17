"""Extract three exact source regions from FD_SH.c for the focused harness."""

from pathlib import Path


def fragments(source, candidate):
    start = source.index("for (nt=1;nt<=8;nt++) acq.srcpos1")
    first = source[start:source.index("/* solve forward problem */", start)]
    reset = source.index("\nnsrc_loc=0;")
    before = source.index("/* Release shot-local source data", start) if candidate else reset + 1
    middle = source[before:source.index("} /* end of loop over shots */", reset)]
    tail = ""
    if not candidate:
        a = source.index("\nif (nsrc_loc>0){", source.index("free_vector(hc"))
        b = source.index(" /* free memory for global source positions */", a)
        tail = source[a+1:b]
    assert first.count("wavelet(") == 1 and first.count("splitsrc(") == 2
    assert middle.rstrip().endswith("nsrc_loc=0;")
    return {"prepare": first, "finish": middle, "after_loop": tail}


def write_fragments(source: Path, destination: Path, *, candidate: bool) -> None:
    destination.mkdir()
    for name, text in fragments(source.read_text(encoding="ascii"), candidate).items():
        (destination / f"{name}.inc").write_text(text, encoding="ascii", newline="\n")
