/*  --------------------------------------------------------------------------
 *   Solving the (visco)-elastic 2D PSV-forward problem by finite-differences 
 *   for a single shot 
 *
 *   mode = 0 - forward modelling only, STF estimation or FWI gradient calculation
 *   mode = 1 - backpropagation of data residuals
 *   mode = 2 - evaluation objective function for step length estimation  
 * 
 *   
 *   D. Koehn
 *   Kiel, 28.12.2015
 *
 *  --------------------------------------------------------------------------*/

#include "fd.h"
#ifdef DENISE_ENABLE_CUDA_PSV
#include "denise_cuda_psv_dispatch.h"
#endif

void visco_psv_exact_step(int t);
extern int NX, NY, FW;

/* Test-only raw observation of the live restart state.  These deliberately
 * traverse the allocated arrays directly instead of sharing any checkpoint
 * packing, restore, or comparison helper. */
static void write_live_matrix(FILE *stream, float **field,
                              int j0, int j1, int i0, int i1) {
	int i, j;
	for (j = j0; j <= j1; ++j)
		for (i = i0; i <= i1; ++i)
			if (fwrite(&field[j][i], sizeof(float), 1, stream) != 1)
				err(" Could not write exact visco PSV live-state test snapshot. ");
}

static void write_live_gsls_l1(FILE *stream, float ***field,
					 int j0, int j1, int i0, int i1) {
	int i, j;
	for (j = j0; j <= j1; ++j)
		for (i = i0; i <= i1; ++i)
			if (fwrite(&field[j][i][1], sizeof(float), 1, stream) != 1)
				err(" Could not write exact visco PSV live-state test snapshot. ");
}

static void write_live_state_snapshot(const char *prefix, const char *label,
						  const struct wavePSV *wave,
						  const struct wavePSV_PML *pml) {
	char path[STRING_SIZE + 96];
	FILE *stream;
	int lo = -2, hi_j = NY + 3, hi_i = NX + 3;

	snprintf(path, sizeof(path), "%s.checkpoint_replay.%s.primary.bin", prefix, label);
	stream = fopen(path, "wb");
	if (!stream) err(" Could not open exact visco PSV primary-state test snapshot. ");
	write_live_matrix(stream, wave->pvx, lo, hi_j, lo, hi_i);
	write_live_matrix(stream, wave->pvy, lo, hi_j, lo, hi_i);
	write_live_matrix(stream, wave->psxx, lo, hi_j, lo, hi_i);
	write_live_matrix(stream, wave->psyy, lo, hi_j, lo, hi_i);
	write_live_matrix(stream, wave->psxy, lo, hi_j, lo, hi_i);
	fclose(stream);

	snprintf(path, sizeof(path), "%s.checkpoint_replay.%s.gsls.bin", prefix, label);
	stream = fopen(path, "wb");
	if (!stream) err(" Could not open exact visco PSV GSLS-state test snapshot. ");
	write_live_gsls_l1(stream, wave->pr, lo, hi_j, lo, hi_i);
	write_live_gsls_l1(stream, wave->pp, lo, hi_j, lo, hi_i);
	write_live_gsls_l1(stream, wave->pq, lo, hi_j, lo, hi_i);
	fclose(stream);

	snprintf(path, sizeof(path), "%s.checkpoint_replay.%s.cpml.bin", prefix, label);
	stream = fopen(path, "wb");
	if (!stream) err(" Could not open exact visco PSV CPML-state test snapshot. ");
	write_live_matrix(stream, pml->psi_sxx_x, 1, NY, 1, 2 * FW);
	write_live_matrix(stream, pml->psi_sxy_x, 1, NY, 1, 2 * FW);
	write_live_matrix(stream, pml->psi_vxx, 1, NY, 1, 2 * FW);
	write_live_matrix(stream, pml->psi_vyx, 1, NY, 1, 2 * FW);
	write_live_matrix(stream, pml->psi_syy_y, 1, 2 * FW, 1, NX);
	write_live_matrix(stream, pml->psi_sxy_y, 1, 2 * FW, 1, NX);
	write_live_matrix(stream, pml->psi_vyy, 1, 2 * FW, 1, NX);
	write_live_matrix(stream, pml->psi_vxy, 1, 2 * FW, 1, NX);
	fclose(stream);
}

static void write_psv_mpi_timing_report(
		const char *path, const double local_timing[5], int mode) {
	extern int MYID_SHOT, NX, NY, NT, FDORDER, L;
	extern int NPROCX, NPROCY, POS[3], BOUNDARY, FREE_SURF;
	extern MPI_Comm SHOT_COMM;
	double local_record[14], *all_records = NULL;
	double component_maxima[5], critical_communication, critical_compute;
	int component_maximum_ranks[5], critical_communication_rank, critical_compute_rank;
	int rank, ranks;
	FILE *report;

	MPI_Comm_size(SHOT_COMM, &ranks);
	for (rank = 0; rank < 5; ++rank) local_record[rank] = local_timing[rank];
	local_record[5] = MYID_SHOT;
	local_record[6] = POS[1];
	local_record[7] = POS[2];
	local_record[8] = NX;
	local_record[9] = NY;
	local_record[10] = (!BOUNDARY && POS[1] == 0);
	local_record[11] = (!BOUNDARY && POS[1] == NPROCX - 1);
	local_record[12] = (POS[2] == 0);
	local_record[13] = (POS[2] == NPROCY - 1);
	if (MYID_SHOT == 0) {
		all_records = malloc((size_t)ranks * 14 * sizeof(double));
		if (!all_records) err(" Out of memory collecting P/SV MPI timing records. ");
	}
	MPI_Gather(local_record, 14, MPI_DOUBLE, all_records, 14, MPI_DOUBLE, 0, SHOT_COMM);
	if (MYID_SHOT != 0) return;
	for (rank = 0; rank < 5; ++rank) {
		component_maxima[rank] = all_records[rank];
		component_maximum_ranks[rank] = (int)all_records[5];
	}
	critical_communication = all_records[2] + all_records[4];
	critical_compute = all_records[1] + all_records[3];
	critical_communication_rank = critical_compute_rank = (int)all_records[5];
	for (rank = 1; rank < ranks; ++rank) {
		const double *record = all_records + 14 * rank;
		int phase;
		for (phase = 0; phase < 5; ++phase) {
			if (record[phase] > component_maxima[phase]) {
				component_maxima[phase] = record[phase];
				component_maximum_ranks[phase] = (int)record[5];
			}
		}
		if (record[2] + record[4] > critical_communication) {
			critical_communication = record[2] + record[4];
			critical_communication_rank = (int)record[5];
		}
		if (record[1] + record[3] > critical_compute) {
			critical_compute = record[1] + record[3];
			critical_compute_rank = (int)record[5];
		}
	}

	report = fopen(path, "w");
	if (!report) err(" Could not open P/SV MPI timing report. ");
	fprintf(report,
		"{\n"
		"  \"schema\": \"denise.psv_mpi_timing.v2\",\n"
		"  \"mode\": %d,\n"
		"  \"ranks\": %d,\n"
		"  \"decomposition\": [%d, %d],\n"
		"  \"global_grid\": [%d, %d],\n"
		"  \"timesteps\": %d,\n"
		"  \"fd_order\": %d,\n"
		"  \"stencil_radius\": %d,\n"
		"  \"L\": %d,\n"
		"  \"free_surface\": %s,\n"
		"  \"boundary_periodic_x\": %s,\n"
		"  \"m8b_fd8_l1_fast_path\": %s,\n"
		"  \"timing_semantics\": {\n"
		"    \"component_wise_rank_maxima\": \"Each phase is maximized independently across ranks; sums need not describe one rank.\",\n"
		"    \"critical_communication\": \"max_r(velocity_exchange_r + stress_exchange_r)\",\n"
		"    \"critical_compute\": \"max_r(velocity_update_r + stress_update_r)\",\n"
		"    \"critical_total\": \"max_r(timestep_loop_r)\"\n"
		"  },\n"
		"  \"timer_resolution_seconds\": %.17g,\n"
		"  \"timer_calls_per_timestep\": 8,\n"
		"  \"barriers_inside_timestep_loop\": 0,\n"
		"  \"component_wise_rank_maxima_seconds\": {\n"
		"    \"timestep_loop\": %.17g,\n"
		"    \"velocity_update\": %.17g,\n"
		"    \"velocity_exchange\": %.17g,\n"
		"    \"stress_update\": %.17g,\n"
		"    \"stress_exchange\": %.17g\n"
		"  },\n"
		"  \"rank_consistent_critical_seconds\": {\n"
		"    \"timestep_loop\": %.17g,\n"
		"    \"communication\": %.17g,\n"
		"    \"compute\": %.17g\n"
		"  },\n"
		"  \"critical_rank_ids\": {\n"
		"    \"timestep_loop\": %d,\n"
		"    \"communication\": %d,\n"
		"    \"compute\": %d,\n"
		"    \"velocity_update\": %d,\n"
		"    \"velocity_exchange\": %d,\n"
		"    \"stress_update\": %d,\n"
		"    \"stress_exchange\": %d\n"
		"  },\n"
		"  \"rank_timings\": [\n",
		mode, ranks, NPROCX, NPROCY, NX * NPROCX, NY * NPROCY, NT,
		FDORDER, FDORDER / 2, L, FREE_SURF ? "true" : "false",
		BOUNDARY ? "true" : "false", (FDORDER == 8 && L == 1) ? "true" : "false",
		MPI_Wtick(),
		component_maxima[0], component_maxima[1], component_maxima[2],
		component_maxima[3], component_maxima[4], component_maxima[0],
		critical_communication, critical_compute,
		component_maximum_ranks[0], critical_communication_rank, critical_compute_rank,
		component_maximum_ranks[1], component_maximum_ranks[2],
		component_maximum_ranks[3], component_maximum_ranks[4]);
	for (rank = 0; rank < ranks; ++rank) {
		const double *record = all_records + 14 * rank;
		fprintf(report,
			"    {\"rank\": %d, \"position\": [%d, %d], \"local_grid\": [%d, %d], "
			"\"left_physical_cpml\": %s, \"right_physical_cpml\": %s, "
			"\"top_physical_boundary\": %s, \"bottom_physical_cpml\": %s, "
			"\"seconds\": {\"timestep_loop\": %.17g, \"velocity_update\": %.17g, "
			"\"velocity_exchange\": %.17g, \"stress_update\": %.17g, "
			"\"stress_exchange\": %.17g, \"communication\": %.17g, "
			"\"compute\": %.17g}}%s\n",
			(int)record[5], (int)record[6], (int)record[7], (int)record[8], (int)record[9],
			record[10] ? "true" : "false", record[11] ? "true" : "false",
			record[12] ? "true" : "false", record[13] ? "true" : "false",
			record[0], record[1], record[2], record[3], record[4],
			record[2] + record[4], record[1] + record[3],
			rank + 1 < ranks ? "," : "");
	}
	fprintf(report, "  ]\n}\n");
	fclose(report);
	free(all_records);
}

void psv(struct wavePSV *wavePSV, struct wavePSV_PML *wavePSV_PML, struct matPSV *matPSV, struct fwiPSV *fwiPSV, struct mpiPSV *mpiPSV,
		 struct seisPSV *seisPSV, struct seisPSVfwi *seisPSVfwi, struct acq *acq, float *hc, int ishot, int nshots, int nsrc_loc,
		 int ns, int ntr, float **Ws, float **Wr, int hin, int *DTINV_help, int mode, MPI_Request *req_send, MPI_Request *req_rec)
{

	/* global variables */
	extern float DT, DH, TSNAP1, TSNAP2, TSNAPINC;
	extern int MYID, MYID_SHOT, FDORDER, FW, L, GRAD_FORM, FC_SPIKE_1, FC_SPIKE_2, ORDER_SPIKE;
	extern int NX, NY, FREE_SURF, BOUNDARY, MODE, QUELLTYP, QUELLTYPB, QUELLART, FDORDER;
	extern int NPROCX, NPROCY, POS[3], NDT, SEISMO, IDXI, IDYI, GRAD_FORM, DTINV;
	extern int SNAP, INVMAT1, INV_STF, EPRECOND, NTDTINV, NXNYI, NT;
	extern char JACOBIAN[STRING_SIZE];
	extern FILE *FP;
	extern MPI_Comm SHOT_COMM;

	/* local variables */
	int i, j, nt, lsamp, lsnap, nsnap, nd, hin1, imat, imat1, imat2, infoout;
	int exact_elastic_psv_adjoint, checkpoint_test, replaying;
	int checkpoint_timestep, replay_first_timestep, replay_last_timestep, replay_steps;
	struct visco_psv_checkpoint *middle_checkpoint;
	float *receiver_reference;
	size_t receiver_compared, receiver_mismatches, operand_compared[6], operand_mismatches[6];
	float tmp, tmp1, muss, lamss;
	const char *timing_path;
	int timing_enabled;
	double timing[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
	double timing_total_start = 0.0, timing_phase_start = 0.0;
#ifdef DENISE_ENABLE_CUDA_PSV
	enum denise_psv_backend psv_backend = DENISE_PSV_BACKEND_CPU;
	if (denise_cuda_psv_backend_preflight(nsrc_loc, ntr, mode, &psv_backend) < 0) {
		char cuda_error[1024];
		snprintf(cuda_error, sizeof(cuda_error), "%s",
			denise_cuda_psv_dispatch_last_error());
		err(cuda_error);
	}
#endif

	nd = FDORDER / 2 + 1;
	exact_elastic_psv_adjoint=((MODE==1)&&(mode==1)&&(L==0)&&
	                            (INVMAT1==1)&&
	                            ((GRAD_FORM==1)||(GRAD_FORM==2)));
	checkpoint_test = replaying = 0;
	checkpoint_timestep = replay_first_timestep = replay_last_timestep = replay_steps = 0;
	middle_checkpoint = NULL;
	receiver_reference = NULL;
	receiver_compared = receiver_mismatches = 0;
	memset(operand_compared, 0, sizeof(operand_compared));
	memset(operand_mismatches, 0, sizeof(operand_mismatches));
	{
		const char *flag = getenv("DENISE_PSV_CHECKPOINT_REPLAY_TEST");
		checkpoint_test = mode == 0 && flag && flag[0] == '1' && flag[1] == '\0';
	}
	if (checkpoint_test)
	{
		if (!visco_psv_exact_supported() || nsrc_loc != 1 || SEISMO != 1 || ntr < 1)
			err(" Exact visco PSV checkpoint replay test requires the frozen one-source exact envelope and receivers. ");
		checkpoint_timestep = NT / 2;
		if (checkpoint_timestep < 1 || checkpoint_timestep >= NT)
			err(" Exact visco PSV checkpoint replay test needs at least three timesteps. ");
		middle_checkpoint = visco_psv_checkpoint_create();
	}

	/*MPI_Barrier(MPI_COMM_WORLD);*/

	if (MYID_SHOT == 0)
	{

		if ((INV_STF == 0) && (mode == 0))
		{
			fprintf(FP, "\n *****  Starting simulation (forward model) for shot %d of %d  ********** \n", ishot, nshots);
		}

		if ((INV_STF == 1) && (mode == 0))
		{
			fprintf(FP, "\n *****  Starting simulation (STF) for shot %d of %d  ********** \n", ishot, nshots);
		}

		if (mode == 1)
		{
			fprintf(FP, "\n *****  Starting simulation (adjoint wavefield)  ********** \n");
		}
	}

	/* initialize PSV wavefields with zero */
	if (L)
	{
		zero_denise_visc_PSV(-nd + 1, NY + nd, -nd + 1, NX + nd, (*wavePSV).pvx, (*wavePSV).pvy, (*wavePSV).psxx, (*wavePSV).psyy, (*wavePSV).psxy,
							 (*wavePSV).ux, (*wavePSV).uy, (*wavePSV).uxy, (*wavePSV).pvxp1, (*wavePSV).pvyp1, (*wavePSV_PML).psi_sxx_x, (*wavePSV_PML).psi_sxy_x,
							 (*wavePSV_PML).psi_vxx, (*wavePSV_PML).psi_vyx, (*wavePSV_PML).psi_syy_y, (*wavePSV_PML).psi_sxy_y, (*wavePSV_PML).psi_vyy, (*wavePSV_PML).psi_vxy,
							 (*wavePSV_PML).psi_vxxs, (*wavePSV).pr, (*wavePSV).pp, (*wavePSV).pq);
	}
	else
	{
		zero_denise_elast_PSV(-nd + 1, NY + nd, -nd + 1, NX + nd, (*wavePSV).pvx, (*wavePSV).pvy, (*wavePSV).psxx, (*wavePSV).psyy, (*wavePSV).psxy,
							  (*wavePSV).ux, (*wavePSV).uy, (*wavePSV).uxy, (*wavePSV).pvxp1, (*wavePSV).pvyp1, (*wavePSV_PML).psi_sxx_x,
							  (*wavePSV_PML).psi_sxy_x, (*wavePSV_PML).psi_vxx, (*wavePSV_PML).psi_vyx, (*wavePSV_PML).psi_syy_y, (*wavePSV_PML).psi_sxy_y,
							  (*wavePSV_PML).psi_vyy, (*wavePSV_PML).psi_vxy, (*wavePSV_PML).psi_vxxs);
	}

#ifdef DENISE_ENABLE_CUDA_PSV
	/* Provisional M8e integration boundary.  The ordinary psv.o is compiled
	 * without this block; only the optional CUDA executable recognizes the
	 * selector and delegates the frozen B2A forward envelope as one resident
	 * invocation.  A requested CUDA path never falls back to this CPU loop. */
	{
		int cuda_dispatch = denise_cuda_psv_dispatch(
			wavePSV, wavePSV_PML, matPSV, seisPSV, acq, hc,
			nsrc_loc, ntr, mode, psv_backend);
		if (cuda_dispatch < 0) {
			char cuda_error[1024];
			snprintf(cuda_error, sizeof(cuda_error), "%s",
				denise_cuda_psv_dispatch_last_error());
			err(cuda_error);
		}
		if (cuda_dispatch > 0)
			return;
	}
#endif

	/*----------------------  loop over timesteps (forward model) ------------------*/

	lsnap = iround(TSNAP1 / DT);
	lsamp = NDT;
	nsnap = 0;

	if (mode == 0)
	{
		hin = 1;
		hin1 = 1;
		imat = 1;
		imat1 = 1;
		imat2 = 1;
	}

	if (mode == 1)
	{
		hin = 1;
		hin1 = 1;
	}

	timing_path = getenv("DENISE_PSV_MPI_TIMING_FILE");
	timing_enabled = mode == 0 && timing_path && timing_path[0] != '\0';
	if (timing_enabled) {
		MPI_Barrier(SHOT_COMM);
		timing_total_start = MPI_Wtime();
	}

	for (nt = 1; nt <= NT; nt++)
	{
		if (mode == 0) visco_psv_exact_step(nt);
		if (replaying)
		{
			if (!replay_steps) replay_first_timestep = nt;
			replay_last_timestep = nt;
			replay_steps++;
		}

		/* Check if simulation is still stable */
		/*if (isnan(pvy[NY/2][NX/2])) err(" Simulation is unstable !");*/
		if (isnan((*wavePSV).pvy[NY / 2][NX / 2]))
		{
			fprintf(FP, "\n Time step: %d; pvy: %f \n", nt, (*wavePSV).pvy[NY / 2][NX / 2]);
			err(" Simulation is unstable !");
		}

		infoout = !(nt % 10000);

		if (MYID_SHOT == 0)
		{
			if (infoout)
				fprintf(FP, "\n Computing timestep %d of %d \n", nt, NT);
			/*time3=MPI_Wtime();*/
		}

		/* update of particle velocities */
		if (timing_enabled) timing_phase_start = MPI_Wtime();
		if (mode == 0 || mode == 2)
		{
			update_v_PML_PSV(1, NX, 1, NY, nt, (*wavePSV).pvx, (*wavePSV).pvxp1, (*wavePSV).pvxm1, (*wavePSV).pvy, (*wavePSV).pvyp1, (*wavePSV).pvym1, (*wavePSV).uttx, (*wavePSV).utty, (*wavePSV).psxx, (*wavePSV).psyy,
							 (*wavePSV).psxy, (*matPSV).prip, (*matPSV).prjp, (*acq).srcpos_loc, (*acq).signals, (*acq).signals, nsrc_loc, (*wavePSV_PML).absorb_coeff, hc, infoout, 0, (*wavePSV_PML).K_x, (*wavePSV_PML).a_x,
							 (*wavePSV_PML).b_x, (*wavePSV_PML).K_x_half, (*wavePSV_PML).a_x_half, (*wavePSV_PML).b_x_half, (*wavePSV_PML).K_y, (*wavePSV_PML).a_y, (*wavePSV_PML).b_y, (*wavePSV_PML).K_y_half,
							 (*wavePSV_PML).a_y_half, (*wavePSV_PML).b_y_half, (*wavePSV_PML).psi_sxx_x, (*wavePSV_PML).psi_syy_y, (*wavePSV_PML).psi_sxy_y, (*wavePSV_PML).psi_sxy_x, 0);
		}

		if (timing_enabled) timing[1] += MPI_Wtime() - timing_phase_start;


                if(mode==1){
	         update_v_PML_PSV(1, NX, 1, NY, nt, (*wavePSV).pvx, (*wavePSV).pvxp1, (*wavePSV).pvxm1, (*wavePSV).pvy, (*wavePSV).pvyp1, (*wavePSV).pvym1, (*wavePSV).uttx, (*wavePSV).utty, (*wavePSV).psxx, (*wavePSV).psyy, 
                              (*wavePSV).psxy, (*matPSV).prip, (*matPSV).prjp, (*acq).srcpos_loc_back, (*seisPSVfwi).sectionvxdiff, (*seisPSVfwi).sectionvydiff,ntr,(*wavePSV_PML).absorb_coeff,hc,infoout, 1, (*wavePSV_PML).K_x,
 	                      (*wavePSV_PML).a_x, (*wavePSV_PML).b_x, (*wavePSV_PML).K_x_half, (*wavePSV_PML).a_x_half, (*wavePSV_PML).b_x_half, (*wavePSV_PML).K_y, (*wavePSV_PML).a_y, (*wavePSV_PML).b_y, (*wavePSV_PML).K_y_half, 
                              (*wavePSV_PML).a_y_half, (*wavePSV_PML).b_y_half, (*wavePSV_PML).psi_sxx_x, (*wavePSV_PML).psi_syy_y, (*wavePSV_PML).psi_sxy_y, (*wavePSV_PML).psi_sxy_x, exact_elastic_psv_adjoint);
                }
		                 
		/*if (MYID==0){
		if (mode == 1)
		{
			update_v_PML_PSV(1, NX, 1, NY, nt, (*wavePSV).pvx, (*wavePSV).pvxp1, (*wavePSV).pvxm1, (*wavePSV).pvy, (*wavePSV).pvyp1, (*wavePSV).pvym1, (*wavePSV).uttx, (*wavePSV).utty, (*wavePSV).psxx, (*wavePSV).psyy,
							 (*wavePSV).psxy, (*matPSV).prip, (*matPSV).prjp, (*acq).srcpos_loc_back, (*seisPSVfwi).sectionvxdiff, (*seisPSVfwi).sectionvydiff, ntr, (*wavePSV_PML).absorb_coeff, hc, infoout, 1, (*wavePSV_PML).K_x,
							 (*wavePSV_PML).a_x, (*wavePSV_PML).b_x, (*wavePSV_PML).K_x_half, (*wavePSV_PML).a_x_half, (*wavePSV_PML).b_x_half, (*wavePSV_PML).K_y, (*wavePSV_PML).a_y, (*wavePSV_PML).b_y, (*wavePSV_PML).K_y_half,
							 (*wavePSV_PML).a_y_half, (*wavePSV_PML).b_y_half, (*wavePSV_PML).psi_sxx_x, (*wavePSV_PML).psi_syy_y, (*wavePSV_PML).psi_sxy_y, (*wavePSV_PML).psi_sxy_x, 0);
		}

		/*if (MYID_SHOT==0){
			time4=MPI_Wtime();
			time_av_v_update+=(time4-time3);
			if (infoout)  fprintf(FP," particle velocity exchange between PEs ...");
		}*/

		/* exchange of particle velocities between PEs */
		if (timing_enabled) timing_phase_start = MPI_Wtime();
		exchange_v_PSV((*wavePSV).pvx, (*wavePSV).pvy, (*mpiPSV).bufferlef_to_rig, (*mpiPSV).bufferrig_to_lef, (*mpiPSV).buffertop_to_bot, (*mpiPSV).bufferbot_to_top, req_send, req_rec);
		if (timing_enabled) timing[2] += MPI_Wtime() - timing_phase_start;

		/* Form 1 needs the B-state velocity multiplier: the time integral
		 * immediately after receiver injection and the reverse V transpose. */
		if(exact_elastic_psv_adjoint && (GRAD_FORM==1)){
			for(i=1;i<=NX;i++){
				for(j=1;j<=NY;j++){
					(*wavePSV).pvxp1[j][i]+=(*wavePSV).pvx[j][i]*DT;
					(*wavePSV).pvyp1[j][i]+=(*wavePSV).pvy[j][i]*DT;
				}
			}
		}

		/* C -> B is applied next.  Therefore PRE stress is the exact material
		 * multiplier and B velocity is the exact inverse-density multiplier. */
		if(exact_elastic_psv_adjoint &&
		   (DTINV_help[NT-nt+1]==1)){
			imat=((NXNYI*NTDTINV)-hin*NXNYI)+1;
			for(i=1;i<=NX;i=i+IDXI){
				for(j=1;j<=NY;j=j+IDYI){
					(*fwiPSV).waveconv_lam_exact[j][i]+=
						((*fwiPSV).forward_prop_x[imat]+
						 (*fwiPSV).forward_prop_y[imat])*
						((*wavePSV).psxx[j][i]+(*wavePSV).psyy[j][i]);
					(*fwiPSV).waveconv_mu_normal_exact[j][i]+=
						((*fwiPSV).forward_prop_x[imat]-
						 (*fwiPSV).forward_prop_y[imat])*
						((*wavePSV).psxx[j][i]-(*wavePSV).psyy[j][i]);
					(*fwiPSV).waveconv_mu_xy_exact[j][i]+=
						(*fwiPSV).forward_prop_u[imat]*(*wavePSV).psxy[j][i];
					if(GRAD_FORM==1){
						(*fwiPSV).waveconv_rho_x_exact[j][i]+=
							(*wavePSV).pvxp1[j][i]*
							(*fwiPSV).forward_prop_rho_x[imat];
						(*fwiPSV).waveconv_rho_y_exact[j][i]+=
							(*wavePSV).pvyp1[j][i]*
							(*fwiPSV).forward_prop_rho_y[imat];
					}else{
						(*fwiPSV).waveconv_rho_x_exact[j][i]+=
							(*wavePSV).pvx[j][i]*
							(*fwiPSV).forward_prop_rho_x[imat];
						(*fwiPSV).waveconv_rho_y_exact[j][i]+=
							(*wavePSV).pvy[j][i]*
							(*fwiPSV).forward_prop_rho_y[imat];
					}
					imat++;
				}
			}
			if(EPRECOND==1) eprecond(Wr,(*wavePSV).pvx,(*wavePSV).pvy);
			hin++;
		}

		/*if (MYID_SHOT==0){
		  time5=MPI_Wtime();
		  time_av_v_exchange+=(time5-time4);
		  if (infoout)  fprintf(FP," finished (real time: %4.2f s).\n",time5-time4);
		}*/

		if (timing_enabled) timing_phase_start = MPI_Wtime();
		if (L) /* viscoelastic */
			update_s_visc_PML_PSV(1, NX, 1, NY, NX, NY, (*wavePSV).pvx, (*wavePSV).pvy, (*wavePSV).ux, (*wavePSV).uy, (*wavePSV).uxy, (*wavePSV).uyx, (*wavePSV).psxx, (*wavePSV).psyy, (*wavePSV).psxy, (*matPSV).ppi, (*matPSV).pu,
								  (*matPSV).puipjp, (*matPSV).prho, hc, infoout, (*wavePSV).pr, (*wavePSV).pp, (*wavePSV).pq, (*matPSV).fipjp, (*matPSV).f, (*matPSV).g, (*matPSV).bip, (*matPSV).bjm, (*matPSV).cip, (*matPSV).cjm,
								  (*matPSV).d, (*matPSV).e, (*matPSV).dip, (*wavePSV_PML).K_x, (*wavePSV_PML).a_x, (*wavePSV_PML).b_x, (*wavePSV_PML).K_x_half, (*wavePSV_PML).a_x_half,
								  (*wavePSV_PML).b_x_half, (*wavePSV_PML).K_y, (*wavePSV_PML).a_y, (*wavePSV_PML).b_y, (*wavePSV_PML).K_y_half, (*wavePSV_PML).a_y_half, (*wavePSV_PML).b_y_half, (*wavePSV_PML).psi_vxx,
								  (*wavePSV_PML).psi_vyy, (*wavePSV_PML).psi_vxy, (*wavePSV_PML).psi_vyx, mode);
		else
			update_s_elastic_PML_PSV(1, NX, 1, NY, NX, NY, (*wavePSV).pvx, (*wavePSV).pvy, (*wavePSV).ux, (*wavePSV).uy, (*wavePSV).uxy, (*wavePSV).uyx, (*wavePSV).psxx, (*wavePSV).psyy, (*wavePSV).psxy, (*matPSV).ppi, (*matPSV).pu,
									 (*matPSV).puipjp, (*wavePSV_PML).absorb_coeff, (*matPSV).prho, hc, infoout, (*wavePSV_PML).K_x, (*wavePSV_PML).a_x, (*wavePSV_PML).b_x, (*wavePSV_PML).K_x_half, (*wavePSV_PML).a_x_half,
									 (*wavePSV_PML).b_x_half, (*wavePSV_PML).K_y, (*wavePSV_PML).a_y, (*wavePSV_PML).b_y, (*wavePSV_PML).K_y_half, (*wavePSV_PML).a_y_half, (*wavePSV_PML).b_y_half, (*wavePSV_PML).psi_vxx,
									 (*wavePSV_PML).psi_vyy, (*wavePSV_PML).psi_vxy, (*wavePSV_PML).psi_vyx, mode);
		if (timing_enabled) timing[3] += MPI_Wtime() - timing_phase_start;

		/* explosive source */
		if (QUELLTYP == 1)
		{

			if (mode == 0 || mode == 2)
			{
				psource(nt, (*wavePSV).psxx, (*wavePSV).psyy, (*acq).srcpos_loc, (*acq).signals, nsrc_loc, 0);
			}
		}

		/* adjoint explosive source */
		if ((QUELLTYPB >= 4) && (mode == 1))
		{
			psource(nt, (*wavePSV).psxx, (*wavePSV).psyy, (*acq).srcpos_loc_back, (*seisPSVfwi).sectionpdiff, nsrc_loc, 1);
		}

		/* moment tensor source */
		if (QUELLTYP == 5)
			msource(nt, (*wavePSV).psxx, (*wavePSV).psyy, (*wavePSV).psxy, (*acq).srcpos_loc, (*acq).signals, nsrc_loc, 0);

		if ((FREE_SURF) && (POS[2] == 0))
		{
			if (L) /* viscoelastic */
				surface_visc_PML_PSV(1, (*wavePSV).pvx, (*wavePSV).pvy, (*wavePSV).psxx, (*wavePSV).psyy, (*wavePSV).psxy, (*wavePSV).pp, (*wavePSV).pq, (*matPSV).ppi,
									 (*matPSV).pu, (*matPSV).prho, (*matPSV).ptaup, (*matPSV).ptaus, (*matPSV).etajm, (*matPSV).peta, hc, (*wavePSV_PML).K_x, (*wavePSV_PML).a_x,
									 (*wavePSV_PML).b_x, (*wavePSV_PML).psi_vxxs);
			else /* elastic */
				surface_elastic_PML_PSV(1, (*wavePSV).pvx, (*wavePSV).pvy, (*wavePSV).psxx, (*wavePSV).psyy, (*wavePSV).psxy, (*matPSV).ppi, (*matPSV).pu, (*matPSV).prho, hc,
										(*wavePSV_PML).K_x, (*wavePSV_PML).a_x, (*wavePSV_PML).b_x, (*wavePSV_PML).psi_vxxs);
		}

		/*if (MYID_SHOT==0){
	      time6=MPI_Wtime();
		  time_av_s_update+=(time6-time5);
	      if (infoout)  fprintf(FP," stress exchange between PEs ...");
	      }*/

		/* stress exchange between PEs */
		if (timing_enabled) timing_phase_start = MPI_Wtime();
		exchange_s_PSV((*wavePSV).psxx, (*wavePSV).psyy, (*wavePSV).psxy,
					   (*mpiPSV).bufferlef_to_rig, (*mpiPSV).bufferrig_to_lef,
					   (*mpiPSV).buffertop_to_bot, (*mpiPSV).bufferbot_to_top,
					   req_send, req_rec);
		if (timing_enabled) timing[4] += MPI_Wtime() - timing_phase_start;

		/*if (MYID_SHOT==0){
	      time7=MPI_Wtime();
	 	  time_av_s_exchange+=(time7-time6);
	     if (infoout)  fprintf(FP," finished (real time: %4.2f s).\n",time7-time6);
	      }  */

		/* store amplitudes at receivers in section-arrays */
		if (SEISMO && (mode == 0 || mode == 2))
		{
			seismo_ssg(nt, ntr, (*acq).recpos_loc, (*seisPSV).sectionvx, (*seisPSV).sectionvy,
					   (*seisPSV).sectionp, (*seisPSV).sectioncurl, (*seisPSV).sectiondiv,
					   (*wavePSV).pvx, (*wavePSV).pvy, (*wavePSV).psxx, (*wavePSV).psyy, (*matPSV).ppi, (*matPSV).pu, (*matPSV).prho, hc);
			if (replaying)
			{
				for (i = 1; i <= ntr; ++i)
				{
					size_t sample = (size_t)(nt - checkpoint_timestep - 1) * ntr + (i - 1);
					float ref_vx = receiver_reference[2 * sample];
					float ref_vy = receiver_reference[2 * sample + 1];
					receiver_compared += 2;
					if (memcmp(&ref_vx, &(*seisPSV).sectionvx[i][nt], sizeof(float)) != 0)
						receiver_mismatches++;
					if (memcmp(&ref_vy, &(*seisPSV).sectionvy[i][nt], sizeof(float)) != 0)
						receiver_mismatches++;
				}
			}
			/*lsamp+=NDT;*/
		}

		/* WRITE SNAPSHOTS TO DISK */
		if ((SNAP) && (nt == lsnap) && (nt <= iround(TSNAP2 / DT)))
		{

			snap(FP, nt, ++nsnap, (*wavePSV).pvx, (*wavePSV).pvy, (*wavePSV).psxx, (*wavePSV).psyy, (*matPSV).pu, (*matPSV).ppi, hc);

			lsnap = lsnap + iround(TSNAPINC / DT);
		}

		/*if (MYID_SHOT==0){
	      time8=MPI_Wtime();
		  time_av_timestep+=(time8-time3);
	      if (infoout)  fprintf(FP," total real time for timestep %d : %4.2f s.\n",nt,time8-time3);
	      } */

		if ((nt == hin1) && (mode == 0) && (MODE > 0) && !replaying)
		{

			/* store forward wavefields for time-domain inversion and RTM */
			/* ---------------------------------------------------------- */

			for (i = 1; i <= NX; i = i + IDXI)
			{
				for (j = 1; j <= NY; j = j + IDYI)
				{
					(*fwiPSV).forward_prop_rho_x[imat1] = (*wavePSV).pvxp1[j][i];
					(*fwiPSV).forward_prop_rho_y[imat1] = (*wavePSV).pvyp1[j][i];
					imat1++;
				}
			}

			for (i = 1; i <= NX; i = i + IDXI)
			{
				for (j = 1; j <= NY; j = j + IDYI)
				{

					/* gradients with data integration */
					if (GRAD_FORM == 1)
					{
						(*fwiPSV).forward_prop_x[imat] = (*wavePSV).psxx[j][i];
						(*fwiPSV).forward_prop_y[imat] = (*wavePSV).psyy[j][i];
					}

					/* gradients without data integration */
					if (GRAD_FORM == 2)
					{
						(*fwiPSV).forward_prop_x[imat] = (*wavePSV).ux[j][i];
						(*fwiPSV).forward_prop_y[imat] = (*wavePSV).uy[j][i];
					}

					imat++;
				}
			}

			for (i = 1; i <= NX; i = i + IDXI)
			{
				for (j = 1; j <= NY; j = j + IDYI)
				{

					/* gradients with data integration */
					if (GRAD_FORM == 1)
					{
						(*fwiPSV).forward_prop_u[imat2] = (*wavePSV).psxy[j][i];
					}

					/* gradients without data integration */
					if (GRAD_FORM == 2)
					{
						(*fwiPSV).forward_prop_u[imat2] = (*wavePSV).uxy[j][i];
					}

					imat2++;
				}
			}

			if ((EPRECOND == 1) || (EPRECOND == 3))
			{
				eprecond(Ws, (*wavePSV).pvx, (*wavePSV).pvy);
			}

			hin++;
			hin1 = hin1 + DTINV;

			DTINV_help[nt] = 1;
		}

		/* save adjoint wavefields for time-domain inversion and partially assemble gradients */
		/* ---------------------------------------------------------------------------------- */
		if ((mode == 1) && (!exact_elastic_psv_adjoint) &&
		    (DTINV_help[NT - nt + 1] == 1))
		{

			imat = ((NXNYI * (NTDTINV)) - hin * NXNYI) + 1;

			for (i = 1; i <= NX; i = i + IDXI)
			{
				for (j = 1; j <= NY; j = j + IDYI)
				{

					(*fwiPSV).waveconv_rho_shot[j][i] += ((*wavePSV).pvxp1[j][i] * (*fwiPSV).forward_prop_rho_x[imat]) + ((*wavePSV).pvyp1[j][i] * (*fwiPSV).forward_prop_rho_y[imat]);

					/* mu-gradient with data integration */
					if (GRAD_FORM == 1)
					{

						(*fwiPSV).waveconv_shot[j][i] += ((*fwiPSV).forward_prop_x[imat] + (*fwiPSV).forward_prop_y[imat]) * ((*wavePSV).psxx[j][i] + (*wavePSV).psyy[j][i]);

						if (INVMAT1 == 1)
						{
							muss = (*matPSV).prho[j][i] * (*matPSV).pu[j][i] * (*matPSV).pu[j][i];
							lamss = (*matPSV).prho[j][i] * (*matPSV).ppi[j][i] * (*matPSV).ppi[j][i] - 2.0 * muss;
						}

						if (INVMAT1 == 3)
						{
							muss = (*matPSV).pu[j][i];
							lamss = (*matPSV).ppi[j][i];
						}

						if (muss > 0.0)
						{
							(*fwiPSV).waveconv_u_shot[j][i] += ((1.0 / (muss * muss)) * ((*fwiPSV).forward_prop_u[imat] * (*wavePSV).psxy[j][i])) + ((1.0 / 4.0) * (((*fwiPSV).forward_prop_x[imat] + (*fwiPSV).forward_prop_y[imat]) * ((*wavePSV).psxx[j][i] + (*wavePSV).psyy[j][i])) / ((lamss + muss) * (lamss + muss))) + ((1.0 / 4.0) * (((*fwiPSV).forward_prop_x[imat] - (*fwiPSV).forward_prop_y[imat]) * ((*wavePSV).psxx[j][i] - (*wavePSV).psyy[j][i])) / (muss * muss));
						}
					}

					/* Vs-gradient without data integration (stress-velocity in non-conservative form) */
					if (GRAD_FORM == 2)
					{

						(*fwiPSV).waveconv_shot[j][i] += ((*fwiPSV).forward_prop_x[imat] + (*fwiPSV).forward_prop_y[imat]) * ((*wavePSV).psxx[j][i] + (*wavePSV).psyy[j][i]);

						if (INVMAT1 == 1)
						{
							muss = (*matPSV).prho[j][i] * (*matPSV).pu[j][i] * (*matPSV).pu[j][i];
							lamss = (*matPSV).prho[j][i] * (*matPSV).ppi[j][i] * (*matPSV).ppi[j][i] - 2.0 * muss;
						}

						if (INVMAT1 == 3)
						{
							muss = (*matPSV).pu[j][i];
							lamss = (*matPSV).ppi[j][i];
						}

						if (muss > 0.0)
						{

							tmp = (1.0 / (4.0 * (lamss + muss) * (lamss + muss))) - (1.0 / (4.0 * muss * muss));
							tmp1 = (1.0 / (4.0 * (lamss + muss) * (lamss + muss))) + (1.0 / (4.0 * muss * muss));

							(*fwiPSV).waveconv_u_shot[j][i] += ((1.0 / (muss * muss)) * ((*fwiPSV).forward_prop_u[imat] * (*wavePSV).psxy[j][i])) + (tmp1 * ((*fwiPSV).forward_prop_x[imat] * (*wavePSV).psxx[j][i] + (*fwiPSV).forward_prop_y[imat] * (*wavePSV).psyy[j][i])) + (tmp * ((*fwiPSV).forward_prop_x[imat] * (*wavePSV).psyy[j][i] + (*fwiPSV).forward_prop_y[imat] * (*wavePSV).psxx[j][i]));
						}
					}

					imat++;
				}
			}

			if (EPRECOND == 1)
			{
				eprecond(Wr, (*wavePSV).pvx, (*wavePSV).pvy);
			}

			hin++;
		}

		/* Canonical M8c boundary: timestep nt is complete, including source,
		 * exchanges, receiver sampling, and the existing trajectory recorder.
		 * Restoring this state resumes with nt+1.  This environment-gated proof
		 * leaves the active exact-FWI lifecycle and recorder unchanged. */
		if (mode == 0)
			visco_psv_exact_forward_boundary(wavePSV, wavePSV_PML, nt);
		if (checkpoint_test && !replaying && nt == checkpoint_timestep)
		{
			write_live_state_snapshot(JACOBIAN, "t300_reference", wavePSV, wavePSV_PML);
			visco_psv_checkpoint_capture(middle_checkpoint, wavePSV, wavePSV_PML, nt);
		}

		if (checkpoint_test && !replaying && nt == NT)
		{
			size_t trace_samples = (size_t)(NT - checkpoint_timestep) * ntr;
			write_live_state_snapshot(JACOBIAN, "t600_reference", wavePSV, wavePSV_PML);
			receiver_reference = malloc(2 * trace_samples * sizeof(float));
			if (!receiver_reference)
				err(" Out of memory retaining exact visco PSV replay traces. ");
			for (j = checkpoint_timestep + 1; j <= NT; ++j)
				for (i = 1; i <= ntr; ++i)
				{
					size_t sample = (size_t)(j - checkpoint_timestep - 1) * ntr + (i - 1);
					receiver_reference[2 * sample] = (*seisPSV).sectionvx[i][j];
					receiver_reference[2 * sample + 1] = (*seisPSV).sectionvy[i][j];
				}
			visco_psv_checkpoint_restore(middle_checkpoint, wavePSV, wavePSV_PML);
			write_live_state_snapshot(JACOBIAN, "t300_restored", wavePSV, wavePSV_PML);
			visco_psv_exact_replay_begin(checkpoint_timestep + 1);
			replaying = 1;
			nt = checkpoint_timestep;
			continue;
		}

		if (checkpoint_test && replaying && nt == NT)
		{
			FILE *report;
			char report_path[STRING_SIZE + 64];
			size_t expected_operands = (size_t)(NT - checkpoint_timestep) * NX * NY;
			size_t expected_receiver_values = 2 * (size_t)(NT - checkpoint_timestep) * ntr;
			int operands_equal = 1, k;
			int timing_equal;
			write_live_state_snapshot(JACOBIAN, "t600_replayed", wavePSV, wavePSV_PML);
			visco_psv_exact_replay_end(operand_compared, operand_mismatches);
			for (k = 0; k < 6; ++k)
				if (operand_compared[k] != expected_operands || operand_mismatches[k])
					operands_equal = 0;
			timing_equal = replay_first_timestep == checkpoint_timestep + 1 &&
				replay_last_timestep == NT && replay_steps == NT - checkpoint_timestep;
			snprintf(report_path, sizeof(report_path), "%s.checkpoint_replay.json", JACOBIAN);
			report = fopen(report_path, "w");
			if (!report) err(" Could not open exact visco PSV checkpoint replay report. ");
			fprintf(report,
				"{\n"
				"  \"checkpoint_timestep\": %d,\n"
				"  \"resume_first_timestep\": %d,\n"
				"  \"resume_last_timestep\": %d,\n"
				"  \"replay_steps\": %d,\n"
				"  \"payload_bytes\": %zu,\n"
				"  \"live_snapshot_full_extent\": [%d, %d],\n"
				"  \"live_snapshot_full_elements_per_field\": %zu,\n"
				"  \"live_snapshot_primary_elements\": %zu,\n"
				"  \"live_snapshot_gsls_elements\": %zu,\n"
				"  \"live_snapshot_cpml_x_extent\": [%d, %d],\n"
				"  \"live_snapshot_cpml_y_extent\": [%d, %d],\n"
				"  \"live_snapshot_cpml_elements\": %zu,\n"
				"  \"operand_compared_per_field\": %zu,\n"
				"  \"operand_compared\": [%zu, %zu, %zu, %zu, %zu, %zu],\n"
				"  \"operand_mismatches\": [%zu, %zu, %zu, %zu, %zu, %zu],\n"
				"  \"receiver_values_compared\": %zu,\n"
				"  \"receiver_mismatches\": %zu,\n"
				"  \"source_timing_equal\": %s,\n"
				"  \"receiver_timing_equal\": %s,\n"
				"  \"bit_identical_replay\": %s\n"
				"}\n",
				checkpoint_timestep, replay_first_timestep, replay_last_timestep,
				replay_steps, visco_psv_checkpoint_payload_bytes(middle_checkpoint),
				NX + 6, NY + 6, (size_t)(NX + 6) * (NY + 6),
				5 * (size_t)(NX + 6) * (NY + 6),
				3 * (size_t)(NX + 6) * (NY + 6),
				NY, 2 * FW, 2 * FW, NX,
				4 * ((size_t)NY * 2 * FW + (size_t)NX * 2 * FW),
				expected_operands,
				operand_compared[0], operand_compared[1], operand_compared[2],
				operand_compared[3], operand_compared[4], operand_compared[5],
				operand_mismatches[0], operand_mismatches[1], operand_mismatches[2],
				operand_mismatches[3], operand_mismatches[4], operand_mismatches[5],
				receiver_compared, receiver_mismatches,
				timing_equal ? "true" : "false",
				(timing_equal && receiver_compared == expected_receiver_values) ? "true" : "false",
				(operands_equal && !receiver_mismatches && timing_equal &&
				 receiver_compared == expected_receiver_values) ? "true" : "false");
			fclose(report);
			free(receiver_reference);
			visco_psv_checkpoint_destroy(middle_checkpoint);
			if (!operands_equal || receiver_mismatches ||
				!timing_equal || receiver_compared != expected_receiver_values)
				err(" Exact visco PSV checkpoint replay was not byte-identical. ");
		}

	} /*--------------------  End  of loop over timesteps ----------*/
	if (timing_enabled) {
		timing[0] = MPI_Wtime() - timing_total_start;
		write_psv_mpi_timing_report(timing_path, timing, mode);
	}
}
