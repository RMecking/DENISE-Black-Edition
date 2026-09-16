/* Focused MPI harness for the actual M6.3c-4 production field maps. */

#include "fd.h"
#include <errno.h>
#include <stdint.h>

int NX, NY, POS[3], NPROCX, NPROCY, BOUNDARY, FDORDER;
int INDEX[5];
const int TAG1 = 1, TAG2 = 2, TAG5 = 5, TAG6 = 6;

struct field {
    float **values;
    float **rows;
    float *data;
    int row_min, row_max, col_min, col_max, row_count, col_count;
};

static struct field allocate_field(int row_min, int row_max, int col_min, int col_max){
    struct field field;
    int row;
    field.row_min = row_min;
    field.row_max = row_max;
    field.col_min = col_min;
    field.col_max = col_max;
    field.row_count = row_max-row_min+1;
    field.col_count = col_max-col_min+1;
    field.rows = (float **)calloc((size_t)field.row_count, sizeof(float *));
    field.data = (float *)calloc(
            (size_t)field.row_count*(size_t)field.col_count, sizeof(float));
    if (!field.rows || !field.data){
        fprintf(stderr, "field allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 92);
    }
    field.values = field.rows-row_min;
    for (row=row_min; row<=row_max; ++row)
        field.values[row] = field.data+(row-row_min)*field.col_count-col_min;
    return field;
}

static void release_field(struct field *field){
    free(field->data);
    free(field->rows);
    field->data = NULL;
    field->rows = NULL;
    field->values = NULL;
}

static float **allocate_buffer(int rows, int columns, float **storage){
    float **pointers;
    float *data;
    int row;
    pointers = (float **)calloc((size_t)rows, sizeof(float *));
    data = (float *)calloc((size_t)rows*(size_t)columns, sizeof(float));
    if (!pointers || !data){
        fprintf(stderr, "buffer allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 93);
    }
    for (row=1; row<=rows; ++row)
        pointers[row-1] = data+(row-1)*columns-1;
    *storage = data;
    return pointers-1;
}

static void release_buffer(float **buffer, float *storage){
    free(storage);
    free(buffer+1);
}

static void copy_flat_to_field(struct field *field, const float *source){
    memcpy(field->data, source,
            (size_t)field->row_count*(size_t)field->col_count*sizeof(float));
}

static double local_dot(const struct field *left, const struct field *right){
    double value = 0.0;
    int index, count = left->row_count*left->col_count;
    for (index=0; index<count; ++index)
        value += (double)left->data[index]*(double)right->data[index];
    return value;
}

static void local_reference_error(
        const struct field *actual, const float *expected,
        double *maximum_difference, double *maximum_reference){
    int index, count = actual->row_count*actual->col_count;
    double difference, reference;
    for (index=0; index<count; ++index){
        difference = fabs((double)actual->data[index]-(double)expected[index]);
        reference = fabs((double)expected[index]);
        if (difference > *maximum_difference) *maximum_difference = difference;
        if (reference > *maximum_reference) *maximum_reference = reference;
    }
}

static void initialize_topology(int rank){
    int ranks = NPROCX*NPROCY;
    POS[1] = rank%NPROCX;
    POS[2] = rank/NPROCX;
    INDEX[1] = rank-1;
    INDEX[2] = rank+1;
    INDEX[3] = rank-NPROCX;
    INDEX[4] = rank+NPROCX;
    if (POS[1] == 0) INDEX[1] += NPROCX;
    if (POS[1] == NPROCX-1) INDEX[2] -= NPROCX;
    if (POS[2] == 0) INDEX[3] = ranks+rank-NPROCX;
    if (POS[2] == NPROCY-1) INDEX[4] = rank+NPROCX-ranks;
}

static int is_velocity(const char *operation){
    return operation[0] == 'v';
}

static int has_exchange(const char *operation){
    return strstr(operation, "exchange") != NULL || strstr(operation, "composed") != NULL;
}

static int has_surface(const char *operation){
    return strstr(operation, "surface") != NULL || strstr(operation, "composed") != NULL;
}

int main(int argc, char **argv){
    struct field input[2], dual[2], forward[2], transpose[2];
    float *payload, *expected_forward[2], *expected_transpose[2];
    float **buffer_left, **buffer_right, **buffer_top, **buffer_bottom;
    float *storage_left, *storage_right, *storage_top, *storage_bottom;
    MPI_Request requests[4];
    int rank, size, field, field_count, fdo, cells, block, status_code;
    int row_min, row_max, col_min, col_max;
    size_t values_per_rank, values_read;
    char filename[4096];
    FILE *stream;
    double lhs_local = 0.0, rhs_local = 0.0, lhs, rhs;
    double diff_local = 0.0, ref_local = 0.0, diff, ref, residual;
    const char *operation, *directory;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc != 9){
        if (rank == 0) fprintf(stderr, "usage: harness npx npy boundary fdorder nx ny operation directory\n");
        MPI_Abort(MPI_COMM_WORLD, 94);
    }
    NPROCX = atoi(argv[1]);
    NPROCY = atoi(argv[2]);
    BOUNDARY = atoi(argv[3]);
    FDORDER = atoi(argv[4]);
    NX = atoi(argv[5]);
    NY = atoi(argv[6]);
    operation = argv[7];
    directory = argv[8];
    if (size != NPROCX*NPROCY || FDORDER < 2 || (FDORDER%2) != 0){
        if (rank == 0) fprintf(stderr, "invalid topology or FD order\n");
        MPI_Abort(MPI_COMM_WORLD, 95);
    }
    initialize_topology(rank);
    field_count = is_velocity(operation) ? 1 : 2;
    fdo = FDORDER/2+1;
    row_min = 1-fdo;
    row_max = NY+fdo;
    col_min = 1-FDORDER/2;
    col_max = NX+FDORDER/2;
    cells = (row_max-row_min+1)*(col_max-col_min+1);
    values_per_rank = (size_t)4*(size_t)field_count*(size_t)cells;
    payload = (float *)malloc(values_per_rank*sizeof(float));
    if (!payload) MPI_Abort(MPI_COMM_WORLD, 96);
    snprintf(filename, sizeof(filename), "%s/rank_%d.bin", directory, rank);
    stream = fopen(filename, "rb");
    if (!stream){
        fprintf(stderr, "rank %d cannot open %s: %s\n", rank, filename, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 97);
    }
    values_read = fread(payload, sizeof(float), values_per_rank, stream);
    fclose(stream);
    if (values_read != values_per_rank){
        fprintf(stderr, "rank %d short input: %lu/%lu\n", rank,
                (unsigned long)values_read, (unsigned long)values_per_rank);
        MPI_Abort(MPI_COMM_WORLD, 98);
    }

    for (field=0; field<field_count; ++field){
        input[field] = allocate_field(row_min, row_max, col_min, col_max);
        dual[field] = allocate_field(row_min, row_max, col_min, col_max);
        forward[field] = allocate_field(row_min, row_max, col_min, col_max);
        transpose[field] = allocate_field(row_min, row_max, col_min, col_max);
        copy_flat_to_field(&input[field], payload+field*cells);
        copy_flat_to_field(&dual[field], payload+(field_count+field)*cells);
        copy_flat_to_field(&forward[field], payload+field*cells);
        copy_flat_to_field(&transpose[field], payload+(field_count+field)*cells);
        block = 2*field_count+field;
        expected_forward[field] = payload+block*cells;
        block = 3*field_count+field;
        expected_transpose[field] = payload+block*cells;
    }

    buffer_left = allocate_buffer(NY, 2*fdo, &storage_left);
    buffer_right = allocate_buffer(NY, 2*fdo, &storage_right);
    buffer_top = allocate_buffer(NX, 2*fdo, &storage_top);
    buffer_bottom = allocate_buffer(NX, 2*fdo, &storage_bottom);

    if (field_count == 1){
        if (has_exchange(operation))
            exchange_v_SH(forward[0].values, buffer_left, buffer_right,
                    buffer_top, buffer_bottom, requests, requests);
        if (has_surface(operation) && POS[2] == 0)
            surface_elastic_SH_velocity(forward[0].values, NX, FDORDER/2);

        if (has_surface(operation) && POS[2] == 0)
            surface_elastic_SH_velocity_adjoint(transpose[0].values, NX, FDORDER/2);
        if (has_exchange(operation)){
            status_code = exchange_v_SH_adjoint(
                    transpose[0].values, NX, NY, FDORDER, BOUNDARY,
                    POS, NPROCX, NPROCY, INDEX, MPI_COMM_WORLD);
            if (status_code != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, status_code);
        }
    } else {
        if (has_surface(operation) && POS[2] == 0)
            surface_elastic_SH_stress(forward[1].values, NX, FDORDER/2);
        if (has_exchange(operation))
            exchange_s_SH(forward[0].values, forward[1].values,
                    buffer_left, buffer_right, buffer_top, buffer_bottom,
                    requests, requests);

        if (has_exchange(operation)){
            status_code = exchange_s_SH_adjoint(
                    transpose[0].values, transpose[1].values,
                    NX, NY, FDORDER, BOUNDARY, POS, NPROCX, NPROCY,
                    INDEX, MPI_COMM_WORLD);
            if (status_code != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, status_code);
        }
        if (has_surface(operation) && POS[2] == 0)
            surface_elastic_SH_stress_adjoint(transpose[1].values, NX, FDORDER/2);
    }

    for (field=0; field<field_count; ++field){
        lhs_local += local_dot(&forward[field], &dual[field]);
        rhs_local += local_dot(&input[field], &transpose[field]);
        local_reference_error(
                &forward[field], expected_forward[field], &diff_local, &ref_local);
        local_reference_error(
                &transpose[field], expected_transpose[field], &diff_local, &ref_local);
    }
    MPI_Reduce(&lhs_local, &lhs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rhs_local, &rhs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&diff_local, &diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ref_local, &ref, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0){
        residual = fabs(lhs-rhs)/fmax(fmax(fabs(lhs), fabs(rhs)), 1.0e-300);
        printf("{\"operation\":\"%s\",\"fdorder\":%d,\"nproc_x\":%d,"
               "\"nproc_y\":%d,\"boundary\":%d,\"lhs\":%.17g,"
               "\"rhs\":%.17g,\"dot_residual\":%.17g,"
               "\"reference_error\":%.17g}\n",
               operation, FDORDER, NPROCX, NPROCY, BOUNDARY, lhs, rhs,
               residual, diff/fmax(ref, 1.0e-300));
        fflush(stdout);
    }

    release_buffer(buffer_left, storage_left);
    release_buffer(buffer_right, storage_right);
    release_buffer(buffer_top, storage_top);
    release_buffer(buffer_bottom, storage_bottom);
    for (field=0; field<field_count; ++field){
        release_field(&input[field]);
        release_field(&dual[field]);
        release_field(&forward[field]);
        release_field(&transpose[field]);
    }
    free(payload);
    MPI_Finalize();
    return 0;
}
