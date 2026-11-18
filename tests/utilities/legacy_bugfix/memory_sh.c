#include "fd.h"

enum { NX_TEST = 12, NY_TEST = 10, NT_TEST = 10, RECEIVERS = 2, FIELD_COUNT = 9 };
int IENDX = NX_TEST, IENDY = NY_TEST, POS[4], REC1, REC2;
int NT = NT_TEST, NDT = 1, QUELLART = 2, QUELLTYPB = 1, QUELLTYP;
int MYID, N_ORDER, GRAD_FORM = 2, FDORDER = 2, ADJ_SIGN = 1, L, FW;
int FREE_SURF, BOUNDARY, NPROCX = 1, NPROCY = 1, INVMAT1 = 3, SEISMO = 1;
float DH = 1.0f, DT = 0.0625f, TS = 1.0f, FC_SPIKE_1, FC_SPIKE_2;
char SIGNAL_FILE[STRING_SIZE] = "memory-fixture";
FILE *FP;

static void need(int passed, const char *why)
{
    if (!passed) {
        fprintf(stderr, "SH fixture: %s\n", why);
        exit(47);
    }
}

static void write_trace(FILE *out, const float *values, size_t count)
{
    need(fwrite(values, sizeof(*values), count, out) == count, "write result");
}

int memory_sh_case(int empty, const char *output)
{
    float **sources = matrix(1, 8, 1, 1);
    int **receivers = imatrix(1, 3, 1, RECEIVERS);
    int *recswitch = ivector(1, RECEIVERS);
    struct seisSHfwi obs = {0};
    float *hc = vector(1, 1);
    int nsrc = -1, nrec = -1;
    float **storage[FIELD_COUNT], **src_loc, **signals;
    int **rec_loc;
    FILE *out = NULL;
    int i, j, field, step;

    FP = stderr;
    if (empty) {
        POS[1] = 1;
        NPROCX = 2;
    }
    hc[1] = 1.0f;
    sources[1][1] = 6; sources[2][1] = 5;
    sources[5][1] = 4; sources[6][1] = 1; sources[8][1] = 1;
    receivers[1][1] = 7; receivers[2][1] = 5;
    receivers[1][2] = 9; receivers[2][2] = 5;

    src_loc = splitsrc(sources, &nsrc, 1);
    rec_loc = splitrec(receivers, &nrec, RECEIVERS, recswitch);
    need(nsrc == (empty ? 0 : 1) && nrec == (empty ? 0 : RECEIVERS), "local ownership");
    need(!empty || (!src_loc && !rec_loc), "empty geometry flags");
    signals = wavelet(src_loc, nsrc, 1);
    alloc_seisSHfwi(nrec, RECEIVERS, NT, &obs);
    need(obs.sectionread && obs.sectionvzdata && obs.sectionvzdiff &&
         obs.sectionvzdiffold, "seismogram storage");

    for (field = 0; field < FIELD_COUNT; ++field)
        storage[field] = matrix(-1, NY_TEST + 2, -1, NX_TEST + 2);
    float **vz = storage[0], **vzp1 = storage[1], **sxz = storage[2],
          **syz = storage[3], **rho = storage[4], **rhoi = storage[5],
          **mu = storage[6], **uz = storage[7], **uzx = storage[8];
    for (j = 1; j <= NY_TEST; ++j)
        for (i = 1; i <= NX_TEST; ++i) {
            rho[j][i] = 2.0f;
            rhoi[j][i] = 0.5f;
            mu[j][i] = 3.0f;
        }

    for (step = 1; step <= NT; ++step) {
        update_v_PML_SH(1, NX_TEST, 1, NY_TEST, step, vz, vzp1, NULL, NULL,
            sxz, syz, rho, rhoi, src_loc, signals, nsrc, NULL, hc, 0, 0, 0,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            NULL, NULL);
        update_s_elastic_PML_SH(1, NX_TEST, 1, NY_TEST, vz, uz, uzx, syz, sxz,
            mu, mu, rho, hc, 0,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            NULL, NULL, 0);
        seismo_ssg(step, nrec, rec_loc, obs.sectionvzdata, obs.sectionvzdata,
                   NULL, NULL, NULL, vz, vz, NULL, NULL, NULL, NULL, NULL, hc);
    }

    if (empty) {
        for (field = 0; field <= 3; ++field)
            for (j = 1; j <= NY_TEST; ++j)
                for (i = 1; i <= NX_TEST; ++i)
                    need(storage[field][j][i] == 0, "empty geometry changed field");
    } else {
        out = fopen(output, "wb");
        need(out != NULL, "open result");
        write_trace(out, &signals[1][1], NT);
        for (i = 1; i <= nrec; ++i)
            write_trace(out, &obs.sectionvzdata[i][1], NT);
        for (field = 0; field <= 3; ++field)
            for (j = 1; j <= NY_TEST; ++j)
                write_trace(out, &storage[field][j][1], NX_TEST);
        need(fclose(out) == 0, "close result");
    }

    for (field = 0; field < FIELD_COUNT; ++field)
        free_matrix(storage[field], -1, NY_TEST + 2, -1, NX_TEST + 2);
    free_matrix(obs.sectionread, 1, RECEIVERS, 1, NT);
    free_matrix(obs.sectionvzdata, 1, nrec, 1, NT);
    free_matrix(obs.sectionvzdiff, 1, nrec, 1, NT);
    free_matrix(obs.sectionvzdiffold, 1, nrec, 1, NT);
    free_matrix(signals, 1, nsrc, 1, NT);
    if (rec_loc)
        free_imatrix(rec_loc, 1, 3, 1, nrec);
    if (src_loc)
        free_matrix(src_loc, 1, 8, 1, nsrc);
    free_ivector(recswitch, 1, RECEIVERS);
    free_imatrix(receivers, 1, 3, 1, RECEIVERS);
    free_matrix(sources, 1, 8, 1, 1);
    free_vector(hc, 1, 1);
    printf("PASS SH %s\n", empty ? "empty" : "positive");
    return 0;
}

#ifdef MEMORY_SH_STANDALONE
int main(int argc, char **argv)
{
    if (argc != 2)
        return 46;
    return memory_sh_case(0, argv[1]);
}
#endif

