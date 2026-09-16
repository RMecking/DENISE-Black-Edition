"""Apply the M6.3b production-adjoint probe only inside a temporary clone."""

from __future__ import annotations

from pathlib import Path


CHANGED_FILES = (
    "src/SH/FWI_SH_visc.c",
    "src/SH/grad_obj_sh_visc.c",
    "src/SH/sh_visc.c",
)


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"M6.3b instrumentation anchor count is not one in {path}: {old!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


def instrument_production_adjoint_probe(repository: Path) -> tuple[str, ...]:
    """Expose dot products without changing the nominal sh_visc mode=1 recurrence."""
    driver = repository / "src/SH/FWI_SH_visc.c"
    old_call = (
        "L2sum = grad_obj_sh(&waveSH, &waveSH_PML, &matSH, &fwiSH, &mpiPSV, "
        "&seisSH, &seisSHfwi, &acq, hc, iter, nsrc, ns, ntr, ntr_glob, \n"
        "nsrc_glob, nsrc_loc, ntr_loc, nstage, We, Ws, Wr, taper_coeff, hin, "
        "DTINV_help, req_send, req_rec);"
    )
    new_call = (
        "L2sum = grad_obj_sh_visc(&waveSH, &waveSH_PML, &matSH, &fwiSH, &mpiPSV, "
        "&seisSH, &seisSHfwi, &acq, hc, iter, nsrc, ns, ntr, ntr_glob,\n"
        "nsrc_glob, nsrc_loc, ntr_loc, nstage, We, Ws, Wr, taper_coeff, hin, "
        "DTINV_help, req_send, req_rec);\n\n"
        "/* M6.3b test-build only: sh_visc has written the probe artifact. */\n"
        "MPI_Finalize();\n"
        "exit(EXIT_SUCCESS);"
    )
    _replace_once(driver, old_call, new_call)

    gradient = repository / "src/SH/grad_obj_sh_visc.c"
    _replace_once(
        gradient,
        '#include "fd.h"\n',
        '#include "fd.h"\n\n'
        "/* M6.3b test-build diagnostic; reduced after the nominal adjoint run. */\n"
        "double M63_PROBE_LEFT_LOCAL = 0.0;\n",
    )
    residual_anchor = (
        "\n\tswstestshot=0;\n\n"
        "\t/* output of time reversed residual seismograms */"
    )
    left_probe = (
        "\n\t/* <F x,y>: calc_res_SH stores the receiver vector in reverse time. */\n"
        "\tM63_PROBE_LEFT_LOCAL = 0.0;\n"
        "\tfor (i=1;i<=ntr;i++){\n"
        "\t   for (j=1;j<=ns;j++){\n"
        "\t      M63_PROBE_LEFT_LOCAL += (double)(*seisSH).sectionvz[i][j]\n"
        "\t                              * (double)(*seisSHfwi).sectionvzdiff[i][ns-j+1];\n"
        "\t   }\n"
        "\t}\n"
    )
    _replace_once(gradient, residual_anchor, left_probe + residual_anchor)

    timestep = repository / "src/SH/sh_visc.c"
    _replace_once(
        timestep,
        "\textern FILE *FP;\n",
        "\textern FILE *FP;\n\textern double M63_PROBE_LEFT_LOCAL;\n",
    )
    _replace_once(
        timestep,
        "\tfloat SUMr, SUMq;\n\n        nd = FDORDER/2 + 1;",
        "\tfloat SUMr, SUMq;\n"
        "\tdouble m63_right_local=0.0, m63_left=0.0, m63_right=0.0;\n"
        "\tconst char *m63_output;\n\n"
        "        nd = FDORDER/2 + 1;\n"
        "\tm63_output = getenv(\"M63_DOT_OUTPUT\");",
    )
    adjoint_anchor = "\tif((mode==1)&&(DTINV_help[NT-nt+1]==1)){"
    right_probe = (
        "\t/* Candidate source-space output of the current nominal mode=1 path.\n"
        "\t * No sign, temporal shift, receiver metric, or scale is fitted here. */\n"
        "\tif(mode==1){\n"
        "\t   for(l=1;l<=nsrc_loc;l++){\n"
        "\t      i=(int)(*acq).srcpos_loc[1][l];\n"
        "\t      j=(int)(*acq).srcpos_loc[2][l];\n"
        "\t      m63_right_local += (double)(*acq).signals[l][NT-nt+1]\n"
        "\t                         * (double)(*waveSH).pvz[j][i];\n"
        "\t   }\n"
        "\t}\n\n"
    )
    _replace_once(timestep, adjoint_anchor, right_probe + adjoint_anchor)
    output_probe = (
        "\tif(mode==1 && m63_output){\n"
        "\t   MPI_Reduce(&M63_PROBE_LEFT_LOCAL,&m63_left,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);\n"
        "\t   MPI_Reduce(&m63_right_local,&m63_right,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);\n"
        "\t   if(MYID==0){\n"
        "\t      FILE *m63_file=fopen(m63_output,\"w\");\n"
        "\t      if(!m63_file){ err(\"M6.3b probe cannot open M63_DOT_OUTPUT\"); }\n"
        "\t      fprintf(m63_file,\n"
        "\t              \"{\\\"left\\\":%.17g,\\\"right\\\":%.17g,\"\n"
        "\t              \"\\\"signed_residual\\\":%.17g,\\\"relative_residual\\\":%.17g}\\n\",\n"
        "\t              m63_left,m63_right,m63_left-m63_right,\n"
        "\t              fabs(m63_left-m63_right)/fmax(fmax(fabs(m63_left),fabs(m63_right)),1.0e-30));\n"
        "\t      if(fclose(m63_file)!=0){ err(\"M6.3b probe cannot close M63_DOT_OUTPUT\"); }\n"
        "\t   }\n"
        "\t   MPI_Barrier(MPI_COMM_WORLD);\n"
        "\t}\n\n"
    )
    text = timestep.read_text(encoding="utf-8")
    if not text.endswith("\n}\n"):
        raise RuntimeError("unexpected sh_visc function terminator")
    timestep.write_text(text[:-3] + "\n" + output_probe + "}\n", encoding="utf-8", newline="\n")
    return CHANGED_FILES
