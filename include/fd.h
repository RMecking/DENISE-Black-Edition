/*------------------------------------------------------------------------
 *  fd.h - header file for DENISE
 *
 *  Daniel Koehn
 *  Kiel, 24.07.2016
 *  ---------------------------------------------------------------------*/

/* files to include */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define iround(x) ((int)(floor)(x+0.5))
#define min(x,y) ((x<y)?x:y)    
#define max(x,y) ((x<y)?y:x)
#define fsign(x) ((x<0.0)?(-1):1)    

#define PI (3.141592653589793)
#define NPAR 100
#define STRING_SIZE 74
#define STRING_SIZE2 256
#define REQUEST_COUNT 4

#define Q_PARAMETERIZATION_LEGACY 0
#define Q_PARAMETERIZATION_PHYSICAL 1

/* Precomputed fixed-FL least-squares map from physical target Q to 1/tau. */
struct q_tau_mapping {
   int mode;
   int sample_count;
   double inverse_tau_per_q;
   double inverse_tau_offset;
};

void init_q_tau_mapping(struct q_tau_mapping *mapping, int mode, int mechanisms,
                        const float *relaxation_frequencies_hz,
                        float fmin_hz, float fmax_hz, float df_hz);
float q_to_tau(float target_q, const struct q_tau_mapping *mapping);
double q_to_tau_derivative(float target_q,
                           const struct q_tau_mapping *mapping);

double visco_sh_harmonic_pair(double left, double right);
int visco_sh_harmonic_pair_vjp(double left, double right, double bar_value,
                               double *bar_left, double *bar_right);
void visco_sh_av_tau_local_vjp(double bar_tau_x, double bar_tau_y,
                               double bar_tau_cells[4]);
double visco_sh_rhoi_value(double rho);
double visco_sh_rhoi_vjp(double rho, double bar_rhoi);
double visco_sh_velocity_rhoi_vjp(double dt, double dh,
                                  double corrected_qx, double corrected_qy,
                                  double bar_v_next);
int visco_sh_material_patch_forward(
        int invmat1, const struct q_tau_mapping *mapping,
        const double primary[4], const double rho[4], const double q[4],
        double output[5]);
int visco_sh_material_patch_vjp(
        int invmat1, const struct q_tau_mapping *mapping,
        const double primary[4], const double rho[4], const double q[4],
        const double bar_output[5], double bar_primary[4],
        double bar_rho[4], double bar_q[4]);

/* C7b consumes cotangents at the outputs of the material-dependent velocity
 * and constitutive operations of one physical timestep.  It returns native
 * coefficient sensitivities only; no time accumulation or physical-model
 * parameter mapping is performed here. */
struct visco_sh_material_timestep_vjp_input {
   int mechanisms;
   double dt, dh;
   double qsum, strain_x, strain_y;
   double bar_v_post_velocity;
   double bar_sxz_next, bar_syz_next;
   const double *bar_r_next, *bar_q_next;
   double mu_x, tau_x, mu_y, tau_y;
   double reference_sum;
   const double *eta_x, *b_x, *eta_y, *b_y;
   double forward_f_x, forward_f_y;
   const double *forward_a_x, *forward_c_x;
   const double *forward_a_y, *forward_c_y;
};

struct visco_sh_material_timestep_vjp_output {
   double g_rhoi;
   double g_mu_x, g_mu_y;
   double g_tau_x, g_tau_y;
};

int visco_sh_material_timestep_vjp(
        const struct visco_sh_material_timestep_vjp_input *input,
        struct visco_sh_material_timestep_vjp_output *output);

/* Exact reduction for the verified discrete objective at DTINV == 1.
 * Operator-level dt factors already belong to the supplied C7b per-step
 * VJPs; the objective is an unweighted sample sum, so this helper adds no
 * temporal scale.  DTINV > 1 has no verified end-to-end contract yet. */
int visco_sh_temporal_native_gradient_accumulate(
        int timesteps, int points, int dtinv,
        const struct visco_sh_material_timestep_vjp_output *series,
        struct visco_sh_material_timestep_vjp_output *accumulated);

struct visco_sh_native_material_gradient_fields {
   double **g_rhoi;
   double **g_mu_x, **g_mu_y;
   double **g_tau_x, **g_tau_y;
};

/* Exact transpose of the locked distributed C6 material graph.  primary_post
 * and rho_post are the post-matcopy fields; owned_q is differentiated only
 * after H^T then V^T has returned tau cotangents to owned cells. */
int visco_sh_distributed_material_gradient_vjp(
        int invmat1, const struct q_tau_mapping *mapping,
        float **primary_post, float **rho_post, float **owned_q,
        const struct visco_sh_native_material_gradient_fields *native,
        float **grad_primary, float **grad_rho, float **grad_q);

/* ---------------------------------- */
/* declaration of PSV data-structures */
/* ---------------------------------- */

/* PSV (visco)-elastic wavefield variables */
struct wavePSV{
   float  ** pvx, ** pvy, **  pvxp1, **  pvyp1, **  pvxm1, **  pvym1;
   float  ** psxx, **  psxy, **  psyy, ** ux, ** uy, ** uxy, ** uyx, ** uttx, ** utty;
   float *** pr, ***pp, ***pq;
} wavePSV; 

/* PSV PML variables*/
struct wavePSV_PML{
   float * d_x, * K_x, * alpha_prime_x, * a_x, * b_x, * d_x_half, * K_x_half, * alpha_prime_x_half; 
   float * a_x_half, * b_x_half, * d_y, * K_y, * alpha_prime_y, * a_y, * b_y, * d_y_half, * K_y_half; 
   float * alpha_prime_y_half, * a_y_half, * b_y_half, ** psi_sxx_x, ** psi_syy_y, ** psi_sxy_y; 
   float ** psi_sxy_x, ** psi_vxx, ** psi_vyy, ** psi_vxy, ** psi_vyx, ** psi_vxxs;
   float  **  absorb_coeff;
} wavePSV_PML;

/* PSV material parameters */
struct matPSV{
   float  **prho, **prip, **prjp, **ppi, **pu, **puipjp;
   float **ptaus, **ptaup, *etaip, *etajm, *peta, **ptausipjp, **fipjp, ***dip, *bip, *bjm;
   float *cip, *cjm, ***d, ***e, **f, **g;
} matPSV;

/* PSV FWI variables */
struct fwiPSV{
   float  **  prho_old, **pu_old, **ppi_old;
   float  ** Vp0, ** Vs0, ** Rho0;
   float  **waveconv, **waveconv_lam, **waveconv_mu, **waveconv_rho, **waveconv_rho_s, **waveconv_u;
   float **waveconv_shot, **waveconv_u_shot, **waveconv_rho_shot;
   /* Native staggered-grid correlations for the exact elastic PSV VJP. */
   float **waveconv_lam_exact, **waveconv_mu_normal_exact, **waveconv_mu_xy_exact;
   float **waveconv_rho_x_exact, **waveconv_rho_y_exact;
   float ** gradg, ** gradp,** gradg_rho, ** gradp_rho, ** gradg_u, ** gradp_u;
   float  *forward_prop_x, *forward_prop_y, *forward_prop_rho_x, *forward_prop_u, *forward_prop_rho_y;
} fwiPSV;

/* PSV seismogram variables */
struct seisPSV{
   float ** sectionvx, ** sectionvy, ** sectionp, ** sectioncurl, ** sectiondiv;
   float ** fulldata, ** fulldata_vx, ** fulldata_vy;
   float ** fulldata_p, ** fulldata_curl,  ** fulldata_div;
} seisPSV;

/* PSV seismogram variables for FWI */
struct seisPSVfwi{
   float ** sectionvxdata, ** sectionvxdiff, ** sectionvxdiffold, ** sectionvydiffold;
   float ** sectionvydiff, ** sectionvydata, ** sectionread;
   float ** sectionpdata, ** sectionpdiff, ** sectionpdiffold;
   float energy;
   double L2;
} seisPSVfwi;

/* Acquisition geometry */
struct acq{
   int * recswitch, ** recpos, ** recpos_loc;
   float ** srcpos, **srcpos_loc, ** srcpos1;
   float ** srcpos_loc_back, ** signals;
} acq;

/* PSV MPI variables */
struct mpiPSV{
   float ** bufferlef_to_rig,  ** bufferrig_to_lef, ** buffertop_to_bot, ** bufferbot_to_top;
} mpiPSV;

/* ---------------------------------- */
/* declaration of VTI data-structures */
/* ---------------------------------- */

/* VTI material parameters */
struct matVTI{
   float  **prho, **prip, **prjp, **c11, **c13, **c33, **c44, **c44h;
} matVTI;

/* ---------------------------------- */
/* declaration of TTI data-structures */
/* ---------------------------------- */

/* TTI material parameters */
struct matTTI{
   float  **prho, **prip, **prjp, **c11, **c13, **c33, **c44, **d11, **d13, **d15, **d33, **d35, **d55, **d15h, **d35h, **d55h, **theta;
} matTTI;

/* ---------------------------------- */
/* declaration of AC data-structures */
/* ---------------------------------- */

/* AC material parameters */
struct matAC{
   float  **prho, **prip, **prjp, **ppi;
   float **ptaus, **ptaup, *etaip, *etajm, *peta, **ptausipjp, **fipjp, ***dip, *bip, *bjm;
   float *cip, *cjm, ***d, ***e, **f, **g;
} matAC;

/* AC (visco)-acoustic wavefield variables */
struct waveAC{
   float  ** pvx, ** pvy, **  pvxp1, **  pvyp1, **  pvxm1, **  pvym1;
   float  ** p, ** ux, ** uy, ** uxy, ** uyx, ** uttx, ** utty;
   float *** pr, ***pp, ***pq;
} waveAC; 

/* AC PML variables*/
struct waveAC_PML{
   float * d_x, * K_x, * alpha_prime_x, * a_x, * b_x, * d_x_half, * K_x_half, * alpha_prime_x_half; 
   float * a_x_half, * b_x_half, * d_y, * K_y, * alpha_prime_y, * a_y, * b_y, * d_y_half, * K_y_half; 
   float * alpha_prime_y_half, * a_y_half, * b_y_half, ** psi_p_x, ** psi_p_y; 
   float ** psi_vxx, ** psi_vyy, ** psi_vxxs;
   float  **  absorb_coeff;
} waveAC_PML;

/* ---------------------------------- */
/* declaration of SH data-structures */
/* ---------------------------------- */

/* SH FWI variables */
struct fwiSH{
   float  **  prho_old, **pu_old, **ptaus_old;
   float  ** Vs0, ** Rho0, ** Taus0;
   float  ** waveconv_mu, **waveconv_rho, **waveconv_ts, **waveconv_rho_s, **waveconv_u,  **waveconv_ts_s;
   float  ** waveconv_u_shot, **waveconv_rho_shot, **waveconv_ts_shot;
   float  ** waveconv_u_x_shot, **waveconv_u_y_shot;
   float  ** gradg_rho, ** gradp_rho, ** gradg_u, ** gradp_u, ** gradg_ts, ** gradp_ts;
   float   * forward_prop_z, *forward_prop_rho_z, *forward_prop_sxz, *forward_prop_syz;
   float  ** forward_prop_rxz, **forward_prop_ryz, ***Rxz, ***Ryz, ***rxzt, ***ryzt;
   float  ** c1mu, ** c4mu, ** c1ts, ** c4ts, * tausl;
   float  ** hess_rho2, ** hess_mu2, ** hess_ts2, **hess_vs2, **hess_rho2p;
   float  ** hess_muts, ** hess_murho, ** hess_tsrho; 
} fwiSH;

/* SH material parameters */
struct matSH{
   float  **prho, **prhoi, **puip, **pujp, **pu, **puipjp;
   float **pqs, **ptaus, *etaip, *etajm, *peta, **ptausipjp, **fipjp, ***dip, *bip, *bjm;
   float *cip, *cjm, ***d, ***e, **f, **g;
} matSH;

/* SH seismogram variables */
struct seisSH{
   float ** sectionvz;
   float ** fulldata_vz;
} seisSH;

/* SH seismogram variables for FWI */
struct seisSHfwi{
   float ** sectionvzdata, ** sectionvzdiff, ** sectionvzdiffold;
   float ** sectionread;
   float energy;
   double L2;
} seisSHfwi;

/* SH (visco)-elastic wavefield variables */
struct waveSH{
   float  ** pvz, **  pvzp1, **  pvzm1, ** psxz, ** psyz;
   float  ** uz, ** uxz, ** uzx, ** uttz;
   float *** pr, ***pp, ***pq;
} waveSH; 

/* SH PML variables*/
struct waveSH_PML{
   float * d_x, * K_x, * alpha_prime_x, * a_x, * b_x, * d_x_half, * K_x_half, * alpha_prime_x_half; 
   float * a_x_half, * b_x_half, * d_y, * K_y, * alpha_prime_y, * a_y, * b_y, * d_y_half, * K_y_half; 
   float * alpha_prime_y_half, * a_y_half, * b_y_half, ** psi_p_x, ** psi_p_y; 
   float ** psi_syz_y, ** psi_sxz_x, ** psi_vzy, ** psi_vzx;
   float  **  absorb_coeff;
} waveSH_PML;

/* Propagating state and fixed coefficients for one exact viscoelastic SH
 * full-state transpose step.  Input and output cotangent states must be
 * distinct.  Rows/columns include the allocated SH halo range; CPML fields
 * use their native NY x 2*FW / 2*FW x NX layouts. */
struct visco_sh_full_state {
   float **vz, **sxz, **syz;
   float ***r, ***q;
   float **psi_sxz_x, **psi_syz_y;
   float **psi_vzx, **psi_vzy;
};

struct visco_sh_full_step_config {
   int nx, ny, fdorder, mechanisms, fw, free_surface, boundary;
   int pos[3], nproc_x, nproc_y, index[5];
   float dt, dh;
   const float *hc;
   float **rhoi, **fipjp, **f;
   const float *bip, *bjm, *cip, *cjm;
   float ***dip, ***d;
   const float *K_x, *a_x, *b_x;
   const float *K_x_half, *a_x_half, *b_x_half;
   const float *K_y, *a_y, *b_y;
   const float *K_y_half, *a_y_half, *b_y_half;
   int nrec;
   const int *rec_x, *rec_y;
   const double *bar_receiver;
   int nsrc;
   const int *src_x, *src_y, *source_type;
   MPI_Comm comm;
};

/* Minimal forward observables required by the exact viscoelastic SH material
 * VJPs.  Each channel is defined only on the owned 1..ny, 1..nx domain. */
struct visco_sh_material_observable_step {
   float **qsum;
   float **strain_x;
   float **strain_y;
};

/* Optional passive material-gradient capture for one full-state adjoint step.
 * Material and observable inputs are read-only.  native_output is overwritten
 * on owned cells only; no temporal accumulation or physical-model mapping is
 * performed by this interface. */
struct visco_sh_material_adjoint_step_context {
   const struct visco_sh_material_observable_step *observable;
   float **mu_x, **tau_x;
   float **mu_y, **tau_y;
   double reference_sum;
   const float *eta_x, *eta_y;
   struct visco_sh_native_material_gradient_fields *native_output;
};

/* Physical-timestep-major storage.  C7 initially requires dtinv == 1. */
struct visco_sh_material_observable_trajectory {
   int nx, ny, nsteps, dtinv;
   struct visco_sh_material_observable_step *steps;
};

/* C7c-b2 material inputs and owned physical-gradient outputs for the
 * reverse-time companion driver.  All inputs are borrowed and read-only;
 * gradient outputs are overwritten only after complete preflight. */
struct visco_sh_reverse_time_material_context {
   const struct visco_sh_material_observable_trajectory *trajectory;
   float **mu_x, **tau_x;
   float **mu_y, **tau_y;
   double reference_sum;
   const float *eta_x, *eta_y;
   int invmat1;
   const struct q_tau_mapping *mapping;
   float **primary_post, **rho_post, **owned_q;
   float **grad_primary, **grad_rho, **grad_q;
};

/* Inactive C8b2-b1 bridge for one already prepared production shot.  The
 * observed traces use the native DENISE [receiver][1..ns] layout.  Gradient
 * outputs are raw owned-cell objective derivatives and are overwritten only
 * after complete preflight. */
struct visco_sh_exact_shot_request {
   struct waveSH *wave;
   struct waveSH_PML *pml;
   struct matSH *material;
   struct fwiSH *fwi;
   struct mpiPSV *mpi;
   struct seisSH *seismogram;
   struct seisSHfwi *legacy_fwi_seismogram;
   struct acq *acquisition;
   float *hc;
   int ishot, nshots, nsrc_local, ns, nrec_local, hin;
   int *dtinv_help;
   float **source_energy, **receiver_energy;
   MPI_Request *request_send, *request_receive;
   float **observed_vz;
   double *receiver_cotangent;
   float **grad_primary, **grad_rho, **grad_q;
};

struct visco_sh_exact_shot_result {
   double objective;
};

/* Inactive C8b2-b2 acquisition/shot wrapper around the locked exact-shot
 * bridge.  For RUN_MULTIPLE_SHOTS != 0, every source column is one shot;
 * otherwise all source columns form one simultaneous-source experiment.
 * Objective and raw owned-cell physical gradients always use this identical
 * shot set.  Input workspaces/acquisition are borrowed; outputs are
 * overwritten after complete preflight. */
struct visco_sh_exact_multi_shot_request {
   struct waveSH *wave;
   struct waveSH_PML *pml;
   struct matSH *material;
   struct fwiSH *fwi;
   struct mpiPSV *mpi;
   struct seisSH *seismogram;
   struct seisSHfwi *legacy_fwi_seismogram;
   struct acq *acquisition;
   float *hc;
   int iter, nsrc, ns, nrec_local, nrec_global, hin;
   int *dtinv_help;
   float **source_energy, **receiver_energy;
   MPI_Request *request_send, *request_receive;
   float **grad_primary, **grad_rho, **grad_q;
};

struct visco_sh_exact_multi_shot_result {
   double objective;
   int shot_count;
};

/* Inactive C8c non-owning optimizer boundary.  Raw inputs are objective
 * derivatives, including physical Q.  Outputs hold the subtractive step p
 * for m_trial = m_base - alpha * p; this baseline adapter uses p = g_raw,
 * so the mathematical trajectory direction is -p. */
struct visco_sh_exact_optimizer_boundary {
   int nx, ny;

   float **grad_raw_primary;
   float **grad_raw_rho;
   float **grad_raw_q;

   float **optimizer_step_primary;
   float **optimizer_step_rho;
   float **optimizer_step_q;
};

/* Inactive C8c non-owning trial-state boundary.  Physical Q is the optimized
 * parameter; tau is derived solver state only. */
struct visco_sh_exact_trial_state_request {
   int nx, ny;

   float alpha;

   int primary_bounds_enabled;
   float primary_lower, primary_upper;
   float rho_lower, rho_upper;
   float q_lower, q_upper;

   const struct q_tau_mapping *q_mapping;

   float **base_primary;
   float **base_rho;
   float **base_q;

   float **optimizer_step_primary;
   float **optimizer_step_rho;
   float **optimizer_step_q;

   float **trial_primary;
   float **trial_rho;
   float **trial_q;
   float **trial_tau;
};

int visco_sh_material_observable_trajectory_init(
        struct visco_sh_material_observable_trajectory *trajectory,
        int nx, int ny, int nsteps, int dtinv, int fw, int free_surface,
        int boundary, int nproc_x, int nproc_y);
void visco_sh_material_observable_trajectory_release(
        struct visco_sh_material_observable_trajectory *trajectory);
int visco_sh_material_observable_begin_step(
        struct visco_sh_material_observable_trajectory *trajectory, int step);
void visco_sh_material_observable_end_step(void);
#if defined(__GNUC__)
int visco_sh_material_observable_is_active(void) __attribute__((weak));
void visco_sh_material_observable_capture_qsum(
        int j, int i, float qsum) __attribute__((weak));
void visco_sh_material_observable_capture_strain(
        int j, int i, float strain_x, float strain_y) __attribute__((weak));
#else
int visco_sh_material_observable_is_active(void);
void visco_sh_material_observable_capture_qsum(int j, int i, float qsum);
void visco_sh_material_observable_capture_strain(
        int j, int i, float strain_x, float strain_y);
#endif
#if defined(M63C_MATERIAL_OBSERVABLE_TEST_COUNTERS)
void visco_sh_material_observable_test_reset_counts(void);
size_t visco_sh_material_observable_test_qsum_count(void);
size_t visco_sh_material_observable_test_strain_count(void);
#endif

/* ------------- */
/* PSV functions */
/* ------------- */

void alloc_fwiPSV(struct fwiPSV *fwiPSV);

void alloc_matPSV(struct matPSV *matPSV);

void alloc_mpiPSV(struct mpiPSV *mpiPSV);

void alloc_seisPSV(int ntr, int ns, struct seisPSV *seisPSV);

void alloc_seisPSVfull(struct seisPSV *seisPSV, int ntr_glob);

void alloc_seisPSVfwi(int ntr, int ntr_glob, int ns, struct seisPSVfwi *seisPSVfwi);

void alloc_PSV(struct wavePSV *wavePSV, struct wavePSV_PML *wavePSV_PML);

void ass_gradPSV(struct fwiPSV *fwiPSV, struct matPSV *matPSV, int iter);

void assemble_gradPSV_exact(struct fwiPSV *fwiPSV, struct matPSV *matPSV,
                            struct mpiPSV *mpiPSV, int iter,
                            MPI_Request *req_send, MPI_Request *req_rec);

float calc_mat_change_test_PSV(float  **  waveconv, float  **  waveconv_rho, float  **  waveconv_u, float  **  rho, float  **  rhonp1, float **  pi, float **  pinp1, float **  u, float **  unp1, 
int iter, int epstest, float eps_scale, int itest);

void calc_res_PSV(struct seisPSV *seisPSV, struct seisPSVfwi *seisPSVfwi, int *recswitch, int  **recpos, int  **recpos_loc, int ntr_glob,  int ntr, int nsrc_glob, float ** srcpos, int ishot, int ns, int iter,
                  int swstestshot);

void dealloc_PSV(struct wavePSV *wavePSV, struct wavePSV_PML *wavePSV_PML);

void exchange_s_PSV(float ** sxx, float ** syy, 
float ** sxy, float ** bufferlef_to_rig, float ** bufferrig_to_lef, 
float ** buffertop_to_bot, float ** bufferbot_to_top,
MPI_Request * req_send, MPI_Request * req_rec);

void exchange_v_PSV(float ** vx, float ** vy,  
float ** bufferlef_to_rig, float ** bufferrig_to_lef, 
float ** buffertop_to_bot, float ** bufferbot_to_top,
MPI_Request * req_send, MPI_Request * req_rec);

void extract_LBFGS_PSV( int iter, float ** waveconv, float ** gradp, float ** waveconv_u, float ** gradp_u, float ** waveconv_rho, float ** gradp_rho, float **ppi, float ** pu, float ** prho, float * r_LBFGS);

void extract_PCG_PSV(float * PCG_old, float ** waveconv, float ** waveconv_u, float ** waveconv_rho);

void FD_PSV();

void FWI_PSV();

double grad_obj_psv(struct wavePSV *wavePSV, struct wavePSV_PML *wavePSV_PML, struct matPSV *matPSV, struct fwiPSV *fwiPSV, struct mpiPSV *mpiPSV, 
         struct seisPSV *seisPSV, struct seisPSVfwi *seisPSVfwi, struct acq *acq, float *hc, int iter, int nsrc, int ns, int ntr, int ntr_glob, int nsrc_glob, 
         int nsrc_loc, int ntr_loc, int nstage, float **We, float **Ws, float **Wr, float ** taper_coeff, int hin, int *DTINV_help, 
         MPI_Request * req_send, MPI_Request * req_rec);

void matcopy_PSV(float ** prho, float ** ppi, float ** pu, float ** ptaup,
float ** ptaus);

void matcopy_elastic_PSV(float ** prho, float ** ppi, float ** pu);

void mem_fwiPSV(int nseismograms,int ntr, int ns, int fdo3, int nd, float buffsize, int ntr_glob);

void mem_PSV(int nseismograms,int ntr, int ns, int fdo3, int nd, float buffsize);

void model_freq_out_PSV(float ** ppi, float  **  rho, float **  pu, int iter, float freq);

void model_it_out_PSV(float ** ppi, float  **  rho, float **  pu, int nstage, int iter, float freq);

double obj_psv(struct wavePSV *wavePSV, struct wavePSV_PML *wavePSV_PML, struct matPSV *matPSV, struct fwiPSV *fwiPSV, struct mpiPSV *mpiPSV, 
struct seisPSV *seisPSV, struct seisPSVfwi *seisPSVfwi, struct acq *acq, float *hc, int nsrc, int nsrc_loc, int nsrc_glob, int ntr, int ntr_glob, 
int ns, int itest, int iter, float **Ws, float **Wr, int hin, int *DTINV_help, float eps_scale, MPI_Request * req_send, MPI_Request * req_rec);

void outseis_PSVfor(struct seisPSV *seisPSV, int *recswitch, int  **recpos, int  **recpos_loc, int ntr_glob, float ** srcpos, int ishot, int ns, int iter, FILE *FP);

void outseis_PSVres(struct seisPSV *seisPSV, struct seisPSVfwi *seisPSVfwi, int *recswitch, int  **recpos, int  **recpos_loc, int ntr_glob, float ** srcpos, int ishot, int ns, int nstage, FILE *FP);

void physics_PSV();

void precond_PSV(struct fwiPSV *fwiPSV, struct acq *acq, int nsrc, int ntr_glob, float ** taper_coeff, FILE *FP_GRAV);

void prepare_update_s_visc_PSV(float *etajm, float *etaip, float *peta, float **fipjp, float **pu,
float **puipjp, float **ppi, float **prho, float **ptaus, float **ptaup,
float **ptausipjp, float **f, float **g, float *bip, float *bjm,
float *cip, float *cjm, float ***dip, float ***d, float ***e);

void psv(struct wavePSV *wavePSV, struct wavePSV_PML *wavePSV_PML, struct matPSV *matPSV, struct fwiPSV *fwiPSV, struct mpiPSV *mpiPSV, 
         struct seisPSV *seisPSV, struct seisPSVfwi *seisPSVfwi, struct acq *acq, float *hc, int ishot, int nshots, int nsrc_loc, 
         int ns, int ntr, float **Ws, float **Wr, int hin, int *DTINV_help, int mode, MPI_Request * req_send, MPI_Request * req_rec);

void readmod_visc_PSV(float  **  rho, float **  pi, float **  u, float **  taus, float **  taup, float *  eta);

void readmod_elastic_PSV(float  **  rho, float **  pi, float **  u);

void RTM_PSV();

void RTM_PSV_out(struct fwiPSV *fwiPSV);

void RTM_PSV_out_shot(struct fwiPSV *fwiPSV, int ishot);

float step_length_est_psv(struct wavePSV *wavePSV, struct wavePSV_PML *wavePSV_PML, struct matPSV *matPSV, struct fwiPSV *fwiPSV, struct mpiPSV *mpiPSV, 
         struct seisPSV *seisPSV, struct seisPSVfwi *seisPSVfwi, struct acq *acq, float *hc, int iter, int nsrc, int ns, int ntr, int ntr_glob, float * epst1, 
         double * L2t, int nsrc_glob, int nsrc_loc, int *step1, int *step3, int nxgrav, int nygrav, int ngrav, float **gravpos, float *gz_mod, int NZGRAV, int ntr_loc, 
         float **Ws, float **Wr, int hin, int *DTINV_help, MPI_Request * req_send, MPI_Request * req_rec);

void stf_psv(struct wavePSV *wavePSV, struct wavePSV_PML *wavePSV_PML, struct matPSV *matPSV, struct fwiPSV *fwiPSV, struct mpiPSV *mpiPSV, struct seisPSV *seisPSV, 
             struct seisPSVfwi *seisPSVfwi, struct acq *acq, float *hc, int ishot, int nshots, int nsrc_loc, int nsrc, int ns, int ntr, int ntr_glob, int iter, float **Ws, 
             float **Wr, int hin, int *DTINV_help, MPI_Request * req_send, MPI_Request * req_rec);

void surface_elastic_PML_PSV(int ndepth, float ** vx, float ** vy, float ** sxx, float ** syy, float ** sxy, float  **  pi, float  **  u, float ** rho, 
			     float * hc, float * K_x, float * a_x, float * b_x, float ** psi_vxx);

void surface_visc_PML_PSV(int ndepth, float ** vx, float ** vy, float ** sxx, float ** syy, float ** sxy, float ***p, float ***q, float  **  ppi, float  **  pu, 
			  float **prho, float **ptaup, float **ptaus, float *etajm, float *peta, float * hc, float * K_x, float * a_x, float * b_x, float ** psi_vxx);

void store_LBFGS_PSV(float ** taper_coeff, int nsrc, float ** srcpos, int ** recpos, int ntr_glob, int iter, float ** waveconv, float ** gradp, float ** waveconv_u, 
float ** gradp_u, float ** waveconv_rho, float ** gradp_rho, float * y_LBFGS, float * s_LBFGS, float * q_LBFGS, float **ppi, float ** pu, float ** prho, int nxnyi, 
int LBFGS_pointer, int NLBFGS, int NLBFGS_vec);

void store_PCG_PSV(float * PCG_old, float ** waveconv, float ** waveconv_u, float ** waveconv_rho);

void update_s_elastic_PML_PSV(int nx1, int nx2, int ny1, int ny2,
float **  vx, float **   vy, float **  ux, float **   uy, float **  uxy, float **   uyx, float **   sxx, float **   syy,
float **   sxy, float ** pi, float ** u, float ** uipjp, float ** absorb_coeff, float **rho, float *hc, int infoout,
float * K_x, float * a_x, float * b_x, float * K_x_half, float * a_x_half, float * b_x_half,
float * K_y, float * a_y, float * b_y, float * K_y_half, float * a_y_half, float * b_y_half,
float ** psi_vxx, float ** psi_vyy, float ** psi_vxy, float ** psi_vyx, int sws);

void update_s_visc_PML_PSV(int nx1, int nx2, int ny1, int ny2,
float **  vx, float **   vy, float **  ux, float **   uy, float **  uxy, float **   uyx, float **   sxx, float **   syy,
float **   sxy, float ** pi, float ** u, float ** uipjp, float **rho, float *hc, int infoout,
float ***r, float ***p, float ***q, float **fipjp, float **f, float **g, float *bip, float *bjm, float *cip, float *cjm, float ***d, float ***e, float ***dip, 
float * K_x, float * a_x, float * b_x, float * K_x_half, float * a_x_half, float * b_x_half,
float * K_y, float * a_y, float * b_y, float * K_y_half, float * a_y_half, float * b_y_half,
float ** psi_vxx, float ** psi_vyy, float ** psi_vxy, float ** psi_vyx, int mode);

void update_v_PML_PSV(int nx1, int nx2, int ny1, int ny2, int nt,
float **  vx, float **  vxp1, float **  vxm1, float ** vy, float **  vyp1, float **  vym1, float **  uttx, float **  utty,float ** sxx, float ** syy,
float ** sxy, float  **rip, float **rjp, float **  srcpos_loc, float ** signals, float ** signals1, int nsrc, float ** absorb_coeff,
float *hc, int infoout,int sw, float * K_x, float * a_x, float * b_x, float * K_x_half, float * a_x_half, float * b_x_half,
float * K_y, float * a_y, float * b_y, float * K_y_half, float * a_y_half, float * b_y_half,
float ** psi_sxx_x, float ** psi_syy_y, float ** psi_sxy_y, float ** psi_syx_x,
int exact_elastic_psv_adjoint);

void zero_denise_elast_PSV(int ny1, int ny2, int nx1, int nx2, float ** vx, float ** vy, float ** sxx, float ** syy, float ** sxy, float ** vxm1, float ** vym1, 
float ** vxym1, float ** vxp1, float ** vyp1, float ** psi_sxx_x, float ** psi_sxy_x, float ** psi_vxx, float ** psi_vyx, float ** psi_syy_y, float ** psi_sxy_y, 
float ** psi_vyy, float ** psi_vxy, float ** psi_vxxs);

void zero_denise_visc_PSV(int ny1, int ny2, int nx1, int nx2, float ** vx, float ** vy, float ** sxx, float ** syy, float ** sxy, float ** vxm1, float ** vym1, 
float ** vxym1, float ** vxp1, float ** vyp1, float ** psi_sxx_x, float ** psi_sxy_x, float ** psi_vxx, float ** psi_vyx, float ** psi_syy_y, float ** psi_sxy_y, 
float ** psi_vyy, float ** psi_vxy, float ** psi_vxxs, float ***pr, float ***pp, float ***pq);

/* ------------- */
/* VTI functions */
/* ------------- */

void alloc_matVTI(struct matVTI *matVTI);

void ass_gradVTI(struct fwiPSV *fwiPSV, struct matVTI *matVTI, int iter);

void checkfd_ssg_VTI(FILE *fp, float ** prho, float ** c11, float ** c13, float ** c33, float ** c44, float *hc);

void FD_VTI();

double grad_obj_VTI(struct wavePSV *wavePSV, struct wavePSV_PML *wavePSV_PML, struct matVTI *matVTI, struct fwiPSV *fwiPSV, struct mpiPSV *mpiPSV, 
         struct seisPSV *seisPSV, struct seisPSVfwi *seisPSVfwi, struct acq *acq, float *hc, int iter, int nsrc, int ns, int ntr, int ntr_glob, int nsrc_glob, 
         int nsrc_loc, int ntr_loc, int nstage, float **We, float **Ws, float **Wr, float ** taper_coeff, int hin, int *DTINV_help, 
         MPI_Request * req_send, MPI_Request * req_rec);

void matcopy_elastic_VTI(float ** rho, float ** pi, float ** u);

void model_elastic_VTI(float  **  rho, float **  c11, float **  c13, float **  c33, float **  c44);

void physics_VTI();

void readmod_elastic_VTI(float  **  rho, float **  c11, float **  c13, float **  c33, float **  c44);

void RTM_VTI();

void seismo_ssg_VTI(int lsamp, int ntr, int **recpos, float **sectionvx, 
float **sectionvy, float **sectionp, float **sectioncurl, float **sectiondiv,
float **vx, float **vy, float **sxx, float **syy, float *hc);

void snap_VTI(FILE *fp,int nt, int nsnap, float **vx, float **vy, float **sxx, float **syy, float *hc);

void update_s_elastic_PML_VTI(int nx1, int nx2, int ny1, int ny2,
	float **  vx, float **   vy, float **  ux, float **   uy, float **  uxy, float **   uyx, float **   sxx, float **   syy,
	float **   sxy, float ** c11,  float ** c13, float ** c33, float ** c44h, float ** absorb_coeff, float *hc, int infoout,
        float * K_x, float * a_x, float * b_x, float * K_x_half, float * a_x_half, float * b_x_half,
        float * K_y, float * a_y, float * b_y, float * K_y_half, float * a_y_half, float * b_y_half,
        float ** psi_vxx, float ** psi_vyy, float ** psi_vxy, float ** psi_vyx, int mode);

void VTI(struct wavePSV *wavePSV, struct wavePSV_PML *wavePSV_PML, struct matVTI *matVTI, struct fwiPSV *fwiPSV, struct mpiPSV *mpiPSV, 
         struct seisPSV *seisPSV, struct seisPSVfwi *seisPSVfwi, struct acq *acq, float *hc, int ishot, int nshots, int nsrc_loc, 
         int ns, int ntr, float **Ws, float **Wr, int hin, int *DTINV_help, int mode, MPI_Request * req_send, MPI_Request * req_rec);

/* ------------- */
/* TTI functions */
/* ------------- */

void alloc_matTTI(struct matTTI *matTTI);

void checkfd_ssg_TTI(FILE *fp, float ** prho, float ** c11, float ** c13, float ** c33, float ** c44, float *hc);

void FD_TTI();

double grad_obj_TTI(struct wavePSV *wavePSV, struct wavePSV_PML *wavePSV_PML, struct matTTI *matTTI, struct fwiPSV *fwiPSV, struct mpiPSV *mpiPSV, 
         struct seisPSV *seisPSV, struct seisPSVfwi *seisPSVfwi, struct acq *acq, float *hc, int iter, int nsrc, int ns, int ntr, int ntr_glob, int nsrc_glob, 
         int nsrc_loc, int ntr_loc, int nstage, float **We, float **Ws, float **Wr, float ** taper_coeff, int hin, int *DTINV_help, 
         MPI_Request * req_send, MPI_Request * req_rec);

void model_elastic_TTI(float  **  rho, float **  c11, float **  c13, float **  c33, float **  c44, float **  theta);

void physics_TTI();

void readmod_elastic_TTI(float  **  rho, float **  c11, float **  c13, float **  c33, float **  c44, float ** theta);

void rot_el_tensor_TTI(struct matTTI *matTTI);

void RTM_TTI();

void update_s_elastic_PML_TTI(int nx1, int nx2, int ny1, int ny2,
	float **  vx, float **   vy, float **  ux, float **   uy, float **  uxy, float **   uyx, float **   sxx, float **   syy,
	float **   sxy, float ** d11,  float ** d13, float ** d15, float ** d15h, float ** d33, float ** d35, float ** d35h, float ** d55h, 
	float ** absorb_coeff, float *hc, int infoout, float * K_x, float * a_x, float * b_x, float * K_x_half, float * a_x_half, float * b_x_half,
        float * K_y, float * a_y, float * b_y, float * K_y_half, float * a_y_half, float * b_y_half,
        float ** psi_vxx, float ** psi_vyy, float ** psi_vxy, float ** psi_vyx, int mode);

void TTI(struct wavePSV *wavePSV, struct wavePSV_PML *wavePSV_PML, struct matTTI *matTTI, struct fwiPSV *fwiPSV, struct mpiPSV *mpiPSV, 
         struct seisPSV *seisPSV, struct seisPSVfwi *seisPSVfwi, struct acq *acq, float *hc, int ishot, int nshots, int nsrc_loc, 
         int ns, int ntr, float **Ws, float **Wr, int hin, int *DTINV_help, int mode, MPI_Request * req_send, MPI_Request * req_rec);

/* ------------- */
/* AC functions  */
/* ------------- */

void ac(struct waveAC *waveAC, struct waveAC_PML *waveAC_PML, struct matAC *matAC, struct fwiPSV *fwiPSV, struct mpiPSV *mpiPSV, 
        struct seisPSV *seisPSV, struct seisPSVfwi *seisPSVfwi, struct acq *acq, float *hc, int ishot, int nshots, int nsrc_loc, 
        int ns, int ntr, float **Ws, float **Wr, int hin, int *DTINV_help, int mode, MPI_Request * req_send, MPI_Request * req_rec);

void alloc_AC(struct waveAC *waveAC, struct waveAC_PML *waveAC_PML);

void alloc_fwiAC(struct fwiPSV *fwiPSV);

void alloc_matAC(struct matAC *matAC);

void ass_gradAC(struct fwiPSV *fwiPSV, struct matAC *matAC, int iter);

float calc_mat_change_test_AC(float  **  waveconv, float  **  waveconv_rho, float  **  rho, float  **  rhonp1, float **  pi, float **  pinp1, int iter, 
                          int epstest, float eps_scale, int itest);

void checkfd_acoustic(FILE *fp, float ** prho, float ** ppi, float *hc);

void dealloc_AC(struct waveAC *waveAC, struct waveAC_PML *waveAC_PML);

void exchange_p_AC(float ** p, float ** bufferlef_to_rig, float ** bufferrig_to_lef, float ** buffertop_to_bot, float ** bufferbot_to_top, 
                   MPI_Request * req_send, MPI_Request * req_rec);

void exchange_v_AC(float ** vx, float ** vy, float ** bufferlef_to_rig, float ** bufferrig_to_lef, float ** buffertop_to_bot, float ** bufferbot_to_top,
	MPI_Request * req_send, MPI_Request * req_rec);

void extract_LBFGS_AC( int iter, float ** waveconv, float ** gradp, float ** waveconv_rho, float ** gradp_rho, float **ppi, float ** prho, float * r_LBFGS);

void extract_PCG_AC(float * PCG_old, float ** waveconv, float ** waveconv_rho);

void FD_AC();

void FWI_AC();

double grad_obj_ac(struct waveAC *waveAC, struct waveAC_PML *waveAC_PML, struct matAC *matAC, struct fwiPSV *fwiPSV, struct mpiPSV *mpiPSV, 
         struct seisPSV *seisPSV, struct seisPSVfwi *seisPSVfwi, struct acq *acq, float *hc, int iter, int nsrc, int ns, int ntr, int ntr_glob, int nsrc_glob, 
         int nsrc_loc, int ntr_loc, int nstage, float **We, float **Ws, float **Wr, float ** taper_coeff, int hin, int *DTINV_help, 
         MPI_Request * req_send, MPI_Request * req_rec);

void matcopy_acoustic_AC(float ** rho, float ** pi);

void mem_fwiAC(int nseismograms, int ntr, int ns, int fdo3, int nd, float buffsize, int ntr_glob);

void model_freq_out_AC(float ** ppi, float  **  rho, int iter, float freq);

void model_it_out_AC(float ** ppi, float  **  rho, int nstage, int iter, float freq);

double obj_ac(struct waveAC *waveAC, struct waveAC_PML *waveAC_PML, struct matAC *matAC, struct fwiPSV *fwiPSV, struct mpiPSV *mpiPSV, 
         struct seisPSV *seisPSV, struct seisPSVfwi *seisPSVfwi, struct acq *acq, float *hc, int nsrc, int nsrc_loc, int nsrc_glob, int ntr, 
         int ntr_glob, int ns, int itest, int iter, float **Ws, float **Wr, int hin, int *DTINV_help, float eps_scale, MPI_Request * req_send, MPI_Request * req_rec);

void physics_AC();

void precond_AC(struct fwiPSV *fwiPSV, struct acq *acq, int nsrc, int ntr_glob, float ** taper_coeff, FILE *FP_GRAV);

void psource_AC(int nt, float ** p, float **  srcpos_loc, float ** signals, int nsrc, int sw);

void readmod_AC(float  **  rho, float **  pi);

void RTM_AC();

void RTM_AC_out(struct fwiPSV *fwiPSV);

void RTM_AC_out_shot(struct fwiPSV *fwiPSV, int ishot);

void seismo_AC(int lsamp, int ntr, int **recpos, float **sectionvx, 
float **sectionvy, float **sectionp, float **sectioncurl, float **sectiondiv,
float **vx, float **vy, float **p, float **pi, float **u, float **rho, float *hc);

void snap_AC(FILE *fp,int nt, int nsnap, float **vx, float **vy, float **p, float **u, float **pi, float *hc);

float step_length_est_ac(struct waveAC *waveAC, struct waveAC_PML *waveAC_PML, struct matAC *matAC, struct fwiPSV *fwiPSV, struct mpiPSV *mpiPSV, 
         struct seisPSV *seisPSV, struct seisPSVfwi *seisPSVfwi, struct acq *acq, float *hc, int iter, int nsrc, int ns, int ntr, int ntr_glob, float * epst1, 
         double * L2t, int nsrc_glob, int nsrc_loc, int *step1, int *step3, int nxgrav, int nygrav, int ngrav, float **gravpos, float *gz_mod, int NZGRAV, int ntr_loc, 
         float **Ws, float **Wr, int hin, int *DTINV_help, MPI_Request * req_send, MPI_Request * req_rec);

void stf_ac(struct waveAC *waveAC, struct waveAC_PML *waveAC_PML, struct matAC *matAC, struct fwiPSV *fwiPSV, struct mpiPSV *mpiPSV, struct seisPSV *seisPSV, 
             struct seisPSVfwi *seisPSVfwi, struct acq *acq, float *hc, int ishot, int nshots, int nsrc_loc, int nsrc, int ns, int ntr, int ntr_glob, int iter, float **Ws, 
             float **Wr, int hin, int *DTINV_help, MPI_Request * req_send, MPI_Request * req_rec);

void store_LBFGS_AC(float ** taper_coeff, int nsrc, float ** srcpos, int ** recpos, int ntr_glob, int iter, float ** waveconv, float ** gradp, float ** waveconv_rho, float ** gradp_rho, float * y_LBFGS, float * s_LBFGS, float * q_LBFGS, float **ppi, float ** prho, int nxnyi, int LBFGS_pointer, int NLBFGS, int NLBFGS_vec);

void store_PCG_AC(float * PCG_old, float ** waveconv, float ** waveconv_rho);

void surface_acoustic_PML_AC(int ndepth, float ** p);

void update_s_acoustic_PML_AC(int nx1, int nx2, int ny1, int ny2,
	float **  vx, float **   vy, float **  ux, float ** p,
	float ** pi, float ** absorb_coeff, float **rho, float *hc, int infoout,
        float * K_x, float * a_x, float * b_x, float * K_x_half, float * a_x_half, float * b_x_half,
        float * K_y, float * a_y, float * b_y, float * K_y_half, float * a_y_half, float * b_y_half,
        float ** psi_vxx, float ** psi_vyy, int mode);

void update_v_PML_AC(int nx1, int nx2, int ny1, int ny2, int nt,
	float **  vx, float **  vxp1, float **  vxm1, float ** vy, float **  vyp1, float **  vym1, float **  uttx, float **  utty, float ** p,
	float  **rip, float **rjp, float **  srcpos_loc, float ** signals, float ** signals1, int nsrc, float ** absorb_coeff,
	float *hc, int infoout,int sw, float * K_x, float * a_x, float * b_x, float * K_x_half, float * a_x_half, float * b_x_half,
        float * K_y, float * a_y, float * b_y, float * K_y_half, float * a_y_half, float * b_y_half,
        float ** psi_p_x, float ** psi_p_y);

void zero_denise_acoustic_AC(int ny1, int ny2, int nx1, int nx2, float ** vx, float ** vy, float ** p, 
                             float ** vxm1, float ** vxp1, float ** vyp1,
                             float ** psi_p_x, float ** psi_vxx, float ** psi_p_y, float ** psi_vyy, 
                             float ** psi_vxxs);

/* ------------- */
/* SH functions  */
/* ------------- */

void alloc_fwiSH(struct fwiSH *fwiSH);

void alloc_matSH(struct matSH *matSH);

void alloc_seisSHfull(struct seisSH *seisSH, int ntr_glob);

void alloc_seisSHfwi(int ntr, int ntr_glob, int ns, struct seisSHfwi *seisSHfwi);

void alloc_seisSH(int ntr, int ns, struct seisSH *seisSH);

void alloc_SH(struct waveSH *waveSH, struct waveSH_PML *waveSH_PML);

void ass_gradSH(struct fwiSH *fwiSH, struct matSH *matSH, int iter);

void assemble_gradSH_exact(struct fwiSH *fwiSH, struct matSH *matSH,
                           struct mpiPSV *mpiPSV, MPI_Request *req_send,
                           MPI_Request *req_rec);

void ass_gradSH_visc(struct fwiSH *fwiSH, struct matSH *matSH, int iter);

void apply_inv_hessSH(struct fwiSH *fwiSH, struct matSH *matSH, int nshots);

void av_mu_SH(float ** u, float ** uip, float ** ujp, float ** rho);

void init_grad_coeff(struct fwiSH *fwiSH, struct matSH *matSH);

void inv_rho_SH(float ** rho, float ** rhoi);

float calc_mat_change_test_SH(float  **  waveconv_rho, float  **  waveconv_u, float  **  rho, 
			      float  **  rhonp1, float **  u, float **  unp1, int iter, int epstest, float eps_scale, int itest);

float calc_mat_change_test_SH_visc(float  **  waveconv_rho, float  **  waveconv_u, float  **  waveconv_ts, float  **  rho, 
			      float  **  rhonp1, float **  u, float **  unp1, float **  ts, float **  tsp1, int iter, 
			      int epstest, float eps_scale, int itest);

void calc_res_SH(struct seisSH *seisSH, struct seisSHfwi *seisSHfwi, int *recswitch, int  **recpos, int  **recpos_loc, int ntr_glob,  int ntr, int nsrc_glob, float ** srcpos, int ishot, int ns, int iter, int swstestshot);

void checkfd_elast_SH(FILE *fp, float ** prho, float ** pu, float *hc);

void checkfd_visc_SH(FILE *fp, float ** prho, float ** pu, float ** ptaus, float *peta, float *hc);

void dealloc_SH(struct waveSH *waveSH, struct waveSH_PML *waveSH_PML);

void eprecond_SH(float ** W, float ** vz);

void exchange_s_SH(float ** sxz, float ** syz, float ** bufferlef_to_rig, float ** bufferrig_to_lef, 
	           float ** buffertop_to_bot, float ** bufferbot_to_top, MPI_Request * req_send, 
		   MPI_Request * req_rec);

void exchange_v_SH(float ** vz, float ** bufferlef_to_rig, float ** bufferrig_to_lef, 
		   float ** buffertop_to_bot, float ** bufferbot_to_top,
	           MPI_Request * req_send, MPI_Request * req_rec);

int visco_sh_exchange_field_adjoint(
        float **bar_field, int nx, int ny, int vertical_depth,
        int horizontal_depth, int boundary, const int pos[3],
        int nproc_x, int nproc_y, const int index[5], MPI_Comm comm);

int exchange_v_SH_adjoint(
        float **bar_vz, int nx, int ny, int fdorder, int boundary,
        const int pos[3], int nproc_x, int nproc_y,
        const int index[5], MPI_Comm comm);

int exchange_s_SH_adjoint(
        float **bar_sxz, float **bar_syz, int nx, int ny, int fdorder,
        int boundary, const int pos[3], int nproc_x, int nproc_y,
        const int index[5], MPI_Comm comm);

void extract_LBFGS_SH( int iter, float ** waveconv_u, float ** gradp_u, float ** waveconv_rho, float ** gradp_rho, float ** pu, float ** prho, float * r_LBFGS);

void extract_LBFGS_SH_visc( int iter, float ** waveconv_u, float ** gradp_u, float ** waveconv_rho, float ** gradp_rho, float ** waveconv_ts, float ** gradp_ts, float ** pu, float ** prho,  float ** ptaus, float * r_LBFGS);

void extract_PCG_SH(float * PCG_old, float ** waveconv_u, float ** waveconv_rho);

void extract_PCG_SH_visc(float * PCG_old, float ** waveconv_u, float ** waveconv_rho, float ** waveconv_ts);

void FD_SH();

void FD_grad_SH();

void FWI_SH();

void FWI_SH_visc();

double grad_obj_sh(struct waveSH *waveSH, struct waveSH_PML *waveSH_PML, struct matSH *matSH, struct fwiSH *fwiSH, struct mpiPSV *mpiPSV, 
         struct seisSH *seisSH, struct seisSHfwi *seisSHfwi, struct acq *acq, float *hc, int iter, int nsrc, int ns, int ntr, int ntr_glob, int nsrc_glob, 
         int nsrc_loc, int ntr_loc, int nstage, float **We, float **Ws, float **Wr, float ** taper_coeff, int hin, int *DTINV_help, 
         MPI_Request * req_send, MPI_Request * req_rec);

double grad_obj_sh_visc(struct waveSH *waveSH, struct waveSH_PML *waveSH_PML, struct matSH *matSH, struct fwiSH *fwiSH, struct mpiPSV *mpiPSV, 
         struct seisSH *seisSH, struct seisSHfwi *seisSHfwi, struct acq *acq, float *hc, int iter, int nsrc, int ns, int ntr, int ntr_glob, int nsrc_glob, 
         int nsrc_loc, int ntr_loc, int nstage, float **We, float **Ws, float **Wr, float ** taper_coeff, int hin, int *DTINV_help, 
         MPI_Request * req_send, MPI_Request * req_rec);

void matcopy_elastic_SH(float ** rho, float ** u);

void matcopy_SH(float ** rho, float ** u, float ** taus);
int matcopy_SH_adjoint(float **bar_rho, float **bar_u, float **bar_taus);

void mem_SH(int nseismograms,int ntr, int ns, int fdo3, int nd, float buffsize);

void model_freq_out_SH(float  **  rho, float **  pu, int iter, float freq);

void model_freq_out_SH_visc(float  **  rho, float **  pu, float ** ptaus, int iter, float freq);

void model_it_out_SH(float  **  rho, float **  pu, int nstage, int iter, float freq);

void model_it_out_SH_visc(float  **  rho, float **  pu, float **  ptaus, int nstage, int iter, float freq);

double obj_sh(struct waveSH *waveSH, struct waveSH_PML *waveSH_PML, struct matSH *matSH, struct fwiSH *fwiSH, struct mpiPSV *mpiPSV, 
         struct seisSH *seisSH, struct seisSHfwi *seisSHfwi, struct acq *acq, float *hc, int nsrc, int nsrc_loc, int nsrc_glob, int ntr, 
         int ntr_glob, int ns, int itest, int iter, float **Ws, float **Wr, int hin, int *DTINV_help, float eps_scale, MPI_Request * req_send, MPI_Request * req_rec);

void outseis_SHfor(struct seisSH *seisSH, int *recswitch, int  **recpos, int  **recpos_loc, int ntr_glob, float ** srcpos, int ishot, int ns, int iter, FILE *FP);

void outseis_SHres(struct seisSH *seisSH, struct seisSHfwi *seisSHfwi, int *recswitch, int  **recpos, int  **recpos_loc, int ntr_glob, float ** srcpos, int ishot, int ns, int nstage, FILE *FP);

void physics_SH();

void PML_pro_SH(float * d_x, float * K_x, float * alpha_prime_x, float * a_x, float * b_x, 
            float * d_x_half, float * K_x_half, float * alpha_prime_x_half, float * a_x_half, float * b_x_half,
            float * d_y, float * K_y, float * alpha_prime_y, float * a_y, float * b_y, 
            float * d_y_half, float * K_y_half, float * alpha_prime_y_half, float * a_y_half, float * b_y_half);

void precond_SH(struct fwiSH *fwiSH, struct acq *acq, int nsrc, int ntr_glob, float ** taper_coeff, FILE *FP_GRAV);

void prepare_update_s_visc_SH(float *etajm, float *etaip, float *peta, float **fipjp, float **pujp, 
		float **puip, float **prho, float **ptaus, float **ptausipjp, float **f, float **g, 
		float *bip, float *bjm, float *cip, float *cjm, float ***dip, float ***d, float ***e);

int visco_sh_gsls_local_derivatives(
        int mechanisms, double dt, double unrelaxed_modulus, double tau,
        double reference_sum, const double *eta, const double *b,
        double *f_tau, double *f_modulus, double *c_tau,
        double *c_modulus);

int visco_sh_gsls_local_vjp(
        int mechanisms, double dt, double strain, double bar_s_next,
        const double *bar_r_next, double forward_f,
        const double *forward_a, const double *forward_c,
        double f_tau, double f_modulus, const double *c_tau,
        const double *c_modulus, double *bar_s_prev,
        double *bar_r_prev, double *bar_strain, double *g_tau,
        double *g_modulus);

int visco_sh_stress_cpml_select_x(
        int i, int nx2, int fw, int boundary, int pos_x, int nproc_x,
        const float *K_x_half, const float *a_x_half,
        const float *b_x_half, int *active, int *aux_index,
        double *K, double *a, double *b);

int visco_sh_stress_cpml_select_y(
        int j, int ny2, int fw, int free_surface, int pos_y, int nproc_y,
        const float *K_y, const float *a_y, const float *b_y,
        const float *K_y_half, const float *a_y_half,
        const float *b_y_half, int *active, int *aux_index,
        double *K, double *a, double *b);

int visco_sh_stress_cpml_local_vjp(
        int active, double K, double a, double b, double bar_e,
        double bar_psi_next, double *bar_e_raw, double *bar_psi_prev);

int visco_sh_stress_spatial_local_vjp(
        int fdorder, double dh, const float *hc, double bar_e_raw_x,
        double bar_e_raw_y, double *bar_vz_patch, int patch_rows,
        int patch_stride, int center_row, int center_col);

int update_s_visc_PML_SH_adjoint_point(
        int fdorder, int mechanisms, double dh, double dt,
        const float *hc, const int cpml_active[2],
        const double cpml_K[2], const double cpml_a[2],
        const double cpml_b[2], const double strain[2],
        const double bar_stress_next[2],
        const double *bar_memory_x_next,
        const double *bar_memory_y_next, const double forward_f[2],
        const double *forward_a_x, const double *forward_a_y,
        const double *forward_c_x, const double *forward_c_y,
        const double f_tau[2], const double f_modulus[2],
        const double *c_tau_x, const double *c_tau_y,
        const double *c_modulus_x, const double *c_modulus_y,
        const double bar_psi_next[2], double bar_stress_prev[2],
        double *bar_memory_x_prev, double *bar_memory_y_prev,
        double bar_psi_prev[2], double *bar_vz_patch, int patch_rows,
        int patch_stride, int center_row, int center_col,
        double g_tau[2], double g_modulus[2]);

int visco_sh_velocity_cpml_select_x(
        int i, int nx2, int fw, int boundary, int pos_x, int nproc_x,
        const float *K_x, const float *a_x, const float *b_x,
        int *active, int *aux_index, double *K, double *a, double *b);

int visco_sh_velocity_cpml_select_y(
        int j, int ny2, int fw, int free_surface, int pos_y, int nproc_y,
        const float *K_y, const float *a_y, const float *b_y,
        int *active, int *aux_index, double *K, double *a, double *b);

int visco_sh_velocity_cpml_local_vjp(
        int active, double K, double a, double b, double bar_q,
        double bar_psi_next, double *bar_d_raw, double *bar_psi_prev);

int visco_sh_velocity_spatial_local_vjp(
        int fdorder, const float *hc, double bar_dx, double bar_dy,
        double *bar_sxz_patch, double *bar_syz_patch, int patch_rows,
        int patch_stride, int center_row, int center_col);

int update_v_PML_SH_adjoint_point(
        int fdorder, double dt, double dh, float rhoi, const float *hc,
        const int cpml_active[2], const double cpml_K[2],
        const double cpml_a[2], const double cpml_b[2],
        double bar_vz_next, const double bar_psi_next[2],
        double *bar_vz_prev, double bar_psi_prev[2],
        double *bar_sxz_patch, double *bar_syz_patch, int patch_rows,
        int patch_stride, int center_row, int center_col);

int visco_sh_receiver_velocity_sampling_vjp(
        int nrec, const int *rec_x, const int *rec_y,
        const double *bar_data, double *bar_vz, int rows, int stride);

int visco_sh_velocity_source_injection_vjp(
        int rows, int stride, const double *bar_vz_after,
        double *bar_vz_before, int nsrc, const int *src_x,
        const int *src_y, const int *source_type, double *bar_signal);

int visco_sh_full_state_adjoint_step(
        const struct visco_sh_full_step_config *config,
        struct visco_sh_full_state *bar_next_work,
        struct visco_sh_full_state *bar_prev,
        double *bar_signal);

int visco_sh_full_state_adjoint_step_material(
        const struct visco_sh_full_step_config *config,
        struct visco_sh_full_state *bar_next_work,
        struct visco_sh_full_state *bar_prev,
        double *bar_signal,
        const struct visco_sh_material_adjoint_step_context *material);

/* Exact reverse-time composition of the fixed-material full-state step.
 * Receiver and source cotangent series are time-major:
 *   bar_receiver_series[n * nrec + receiver]
 *   bar_signal_series[n * nsrc + source]
 * for chronological forward indices n = 0, ..., nsteps - 1.  The terminal
 * and scratch states are mutable workspaces; bar_initial is overwritten and
 * owns the result for every positive nsteps, independent of parity. */
int visco_sh_reverse_time_adjoint(
        const struct visco_sh_full_step_config *base_config,
        int nsteps,
        const double *bar_receiver_series,
        struct visco_sh_full_state *bar_terminal_work,
        struct visco_sh_full_state *bar_initial,
        struct visco_sh_full_state *scratch,
        double *bar_signal_series);

/* Material-aware C7c-b2 companion.  It preserves the locked fixed-material
 * state transpose, uses trajectory->steps[n] at reverse step n, accumulates
 * native contributions without weight, then applies the locked C7c-a
 * direct discrete-time sum and distributed physical mapping exactly once.
 * The integrated path is intentionally restricted to trajectory dtinv == 1. */
int visco_sh_reverse_time_adjoint_material(
        const struct visco_sh_full_step_config *base_config,
        int nsteps,
        const double *bar_receiver_series,
        struct visco_sh_full_state *bar_terminal_work,
        struct visco_sh_full_state *bar_initial,
        struct visco_sh_full_state *scratch,
        double *bar_signal_series,
        const struct visco_sh_reverse_time_material_context *material);

int visco_sh_exact_objective_gradient_shot(
        const struct visco_sh_exact_shot_request *request,
        struct visco_sh_exact_shot_result *result);

int visco_sh_exact_objective_gradient(
        const struct visco_sh_exact_multi_shot_request *request,
        struct visco_sh_exact_multi_shot_result *result);

int visco_sh_exact_build_steepest_subtractive_step(
        const struct visco_sh_exact_optimizer_boundary *boundary);

int visco_sh_exact_build_trial_parameter_state(
        const struct visco_sh_exact_trial_state_request *request);

void readmod_elastic_SH(float  **rho, float **u);

void readmod_visc_SH(float **rho, float **u, float **qs, float **taus, float *eta);

void RTM_SH_out_shot(struct fwiSH *fwiSH, int ishot);

void saveseis_glob_SH(FILE *fp, float **sectionvz, int  **recpos, int  **recpos_loc, int ntr, float ** srcpos, int ishot, int ns, int iter);

void sh(struct waveSH *waveSH, struct waveSH_PML *waveSH_PML, struct matSH *matSH, struct fwiSH *fwiSH, struct mpiPSV *mpiPSV, 
         struct seisSH *seisSH, struct seisSHfwi *seisSHfwi, struct acq *acq, float *hc, int ishot, int nshots, int nsrc_loc, 
         int ns, int ntr, float **Ws, float **Wr, int hin, int *DTINV_help, int mode, MPI_Request * req_send, MPI_Request * req_rec);	  

void sh_visc(struct waveSH *waveSH, struct waveSH_PML *waveSH_PML, struct matSH *matSH, struct fwiSH *fwiSH, struct mpiPSV *mpiPSV, 
         struct seisSH *seisSH, struct seisSHfwi *seisSHfwi, struct acq *acq, float *hc, int ishot, int nshots, int nsrc_loc, 
         int ns, int ntr, float **Ws, float **Wr, int hin, int *DTINV_help, int mode, MPI_Request * req_send, MPI_Request * req_rec);	  

void sh_visc_with_material_trajectory(struct waveSH *waveSH, struct waveSH_PML *waveSH_PML, struct matSH *matSH, struct fwiSH *fwiSH, struct mpiPSV *mpiPSV,
         struct seisSH *seisSH, struct seisSHfwi *seisSHfwi, struct acq *acq, float *hc, int ishot, int nshots, int nsrc_loc,
         int ns, int ntr, float **Ws, float **Wr, int hin, int *DTINV_help, int mode, MPI_Request * req_send, MPI_Request * req_rec,
         struct visco_sh_material_observable_trajectory *trajectory);


float step_length_est_sh(struct waveSH *waveSH, struct waveSH_PML *waveSH_PML, struct matSH *matSH, struct fwiSH *fwiSH, struct mpiPSV *mpiPSV, 
         struct seisSH *seisSH, struct seisSHfwi *seisSHfwi, struct acq *acq, float *hc, int iter, int nsrc, int ns, int ntr, int ntr_glob, float * epst1, 
         double * L2t, int nsrc_glob, int nsrc_loc, int *step1, int *step3, int nxgrav, int nygrav, int ngrav, float **gravpos, float *gz_mod, int NZGRAV, int ntr_loc, 
         float **Ws, float **Wr, int hin, int *DTINV_help, MPI_Request * req_send, MPI_Request * req_rec);

void stf_sh(struct waveSH *waveSH, struct waveSH_PML *waveSH_PML, struct matSH *matSH, struct fwiSH *fwiSH, struct mpiPSV *mpiPSV, struct seisSH *seisSH, 
             struct seisSHfwi *seisSHfwi, struct acq *acq, float *hc, int ishot, int nshots, int nsrc_loc, int nsrc, int ns, int ntr, int ntr_glob, int iter, float **Ws, 
             float **Wr, int hin, int *DTINV_help, MPI_Request * req_send, MPI_Request * req_rec);

void store_LBFGS_SH(float ** taper_coeff, int nsrc, float ** srcpos, int ** recpos, int ntr_glob, int iter, float ** waveconv_u, float ** gradp_u, float ** waveconv_rho, 
		    float ** gradp_rho, float * y_LBFGS, float * s_LBFGS, float * q_LBFGS, float ** pu, float ** prho, int nxnyi, int LBFGS_pointer, int NLBFGS, int NLBFGS_vec);

void store_LBFGS_SH_visc(float ** taper_coeff, int nsrc, float ** srcpos, int ** recpos, int ntr_glob, int iter, float ** waveconv_u, float ** gradp_u, float ** waveconv_rho, 
                    float ** gradp_rho, float ** waveconv_ts, float ** gradp_ts, float * y_LBFGS, float * s_LBFGS, float * q_LBFGS, float ** pu, float ** prho, float **ptaus, 
		    int nxnyi, int LBFGS_pointer, int NLBFGS, int NLBFGS_vec);

void store_PCG_SH(float * PCG_old, float ** waveconv_u, float ** waveconv_rho);

void store_PCG_SH_visc(float * PCG_old, float ** waveconv_u, float ** waveconv_rho, float ** waveconv_ts);

void store_pseudo_hess_SH(struct fwiSH *fwiSH);

void surface_elastic_SH_velocity(float **vz, int nx, int half_order);

void surface_elastic_SH_stress(float **syz, int nx, int half_order);

void surface_elastic_SH_velocity_adjoint(
        float **bar_vz, int nx, int half_order);

void surface_elastic_SH_stress_adjoint(
        float **bar_syz, int nx, int half_order);

void update_s_elastic_PML_SH(int nx1, int nx2, int ny1, int ny2,
	float ** vz, float **  uz, float **  uzx, float **   syz, float **   sxz,
	float ** ujp, float ** uip, float **rho, float *hc, int infoout,
        float * K_x, float * a_x, float * b_x, float * K_x_half, float * a_x_half, float * b_x_half,
        float * K_y, float * a_y, float * b_y, float * K_y_half, float * a_y_half, float * b_y_half,
        float ** psi_vzx, float ** psi_vzy, int mode);

void update_s_visc_PML_SH(int nx1, int nx2, int ny1, int ny2,
	float ** vz, float **  uz, float **  uzx, float **   syz, float **   sxz,
	float ** ujp, float ** uip, float **rho, float *hc, int infoout,
	float ***r, float ***p, float ***q, float **fipjp, float **f, float **g, float *bip, float *bjm, 
	float *cip, float *cjm, float ***d, float ***e, float ***dip, 
        float * K_x, float * a_x, float * b_x, float * K_x_half, float * a_x_half, float * b_x_half,
        float * K_y, float * a_y, float * b_y, float * K_y_half, float * a_y_half, float * b_y_half,
        float ** psi_vzx, float ** psi_vzy, struct fwiSH *fwiSH, int mode);

void update_v_PML_SH(int nx1, int nx2, int ny1, int ny2, int nt,
	float **  vz, float **  vzp1, float **  vzm1, float **  utty,float ** sxz, float ** syz,
	float  **rho, float **rhoi, float **  srcpos_loc, float ** signals, int nsrc, float ** absorb_coeff,
	float *hc, int infoout,int sw, int exact_elastic_sh_adjoint, float * K_x, float * a_x, float * b_x, float * K_x_half, float * a_x_half,
	float * b_x_half, float * K_y, float * a_y, float * b_y, float * K_y_half, float * a_y_half, 
	float * b_y_half, float ** psi_sxz_x, float ** psi_syz_y);

void zero_denise_elast_SH(int ny1, int ny2, int nx1, int nx2, float ** vz, float ** sxz, float ** syz, float ** vzm1, 
			 float ** vzp1, float ** psi_sxz_x, float ** psi_syz_y, float ** psi_vzx,  float ** psi_vzy);

void zero_denise_visc_SH(int ny1, int ny2, int nx1, int nx2, float ** vz, float ** sxz, float ** syz, float ** vzm1, 
			 float ** vzp1, float ** psi_sxz_x, float ** psi_syz_y, float ** psi_vzx,  float ** psi_vzy, 
                         float ***pr, float ***pp, float ***pq, float ***Rxz, float ***Ryz);

/* ----------------- */
/* General functions */
/* ----------------- */

void mat_inv_3x3(float **A, float **Ainv);

void window_cos(float **win, int npad, int nsrc, float it1, float it2, float it3, float it4);

void catseis(float **data, float **fulldata, int *recswitch, int ntr_glob, MPI_Comm newcomm_nodentr);

int **splitrec(int **recpos,int *ntr_loc, int ntr, int *recswitch);

void absorb(float ** absorb_coeff);

void smooth_model(float ** pinp1, float ** unp1, float ** rho, int iter);

void taper_grad(float ** waveconv, float ** taper_coeff, float **srcpos, int nshots, int **recpos, int ntr, int sws);

void taper_grad_shot(float ** waveconv,float ** taper_coeff, float **srcpos, int nshots, int **recpos, int ntr, int ishot);

void spat_filt(float ** waveconv, int iter, int sws);

void apply_tdfilt(float **section, int ntr, int ns, int order, float fc2, float fc1);

void av_harm(float ** m, float ** mh);

void av_mat(float **  pi, float **  u, 
float **  ppijm, float **  puip, float ** pujm);

void av_mue(float ** u, float ** uipjp, float ** rho);

void av_rho(float **rho, float **rip, float **rjp);

void av_tau(float **taus, float **tausipjp);

float median2d(float **mat, int ny, int nx);

void calc_envelope(float ** datatrace, float ** envelope, int ns, int ntr);

void calc_hilbert(float ** datatrace, float ** envelope, int ns, int ntr);

double calc_res(float **sectiondata, float **section, float **sectiondiff, float **sectiondiffold, int ntr, int ns, int LNORM, double L2, int itest, int sws, int swstestshot, int ntr_glob, int **recpos, int **recpos_loc, float **srcpos, int nsrc_glob, int ishot, int iter);

double calc_res_grav(int ngrav, float *gz_mod, float *gz_res);

float calc_opt_step(double *  L2t, float * epst, int sws);

double calc_energy(float **sectiondata, int ntr, int ns, float energy, int ntr_glob, int **recpos_loc, int nsrc_glob, int ishot);

void checkfd_ssg_elastic(FILE *fp, float ** prho, float ** ppi, float ** pu, float *hc);

void checkfd_ssg_visc(FILE *fp, float ** prho, float ** ppi, float ** pu, float ** ptaus, float ** ptaup, float * peta, float *hc);

void check_mode_phys();

void comm_ini(float ** bufferlef_to_rig, float ** bufferrig_to_lef, 
float ** buffertop_to_bot, float ** bufferbot_to_top, 
MPI_Request *req_send, MPI_Request *req_rec);

void conv_FD(float * temp_TS, float * temp_TS1, float * temp_conv, int ns);

void copy_mat(float ** A, float ** B);

void count_src();

void descent(float ** grad, float ** gradm);

float dotp(float * vec1, float *vec2, int n1, int n2, int sw);

void eprecond(float ** W, float ** vx, float ** vy);

void eprecond1(float ** We, float ** Ws, float ** Wr);

void extend_mod(float  **rho_grav, float  **rho_grav_ext, int nxgrav, int nygrav);

void gauss_filt(float ** waveconv);

void gauss_filt_var(float ** waveconv, float ** vel_mod);

void grav_grad(int ngrav, float **gravpos, float **grad_grav, float *gz_res);

void grav_mod(float  **rho, int ngrav, float **gravpos, float *gz, int NXGRAV, int NYGRAV, int NZGRAV);

void read_back_density(float ** rho_back);

float *holbergcoeff(void);

int householder(int m, int n, float **mat, float *b);

void info(FILE *fp);

void init_grad(float ** A);

void initproc(void);

void interpol(int ni1, int ni2, float **  intvar, int cfgt_check);

void LBFGS(int iter, float * y_LBFGS, float * s_LBFGS, float * rho_LBFGS, float * alpha_LBFGS, float * q_LBFGS, float * r_LBFGS, float * beta_LBFGS, int LBFGS_pointer, int NLBFGS, int NLBFGS_vec);
           
double LU_decomp(double  **A, double *x, double *b,int n);

float maximum_m(float **mat, int nx, int ny);

void median_model(float ** waveconv, int filt_size);

void median_src(float ** waveconv,float ** taper_coeff, float **srcpos, int nshots, int **recpos, int ntr, int iter, int sws);

float minimum_m(float **mat, int nx, int ny);

void model(float  **  rho, float **  pi, float **  u, float **  taus, float **  taup, float *  eta);

void model_elastic(float  **  rho, float **  pi, float **  u);
			  
void merge(int nsnap, int type);

void mergemod(char modfile[STRING_SIZE], int format);

void msource(int nt, float ** sxx, float ** syy, float ** sxy, float **  srcpos_loc, float ** signals, int nsrc, int sw);

void norm(float **waveconv);

void note(FILE *fp);

void  outseis(FILE *fp, FILE *fpdata, int comp, float **section,
int **recpos, int **recpos_loc, int ntr, float ** srcpos_loc,
int nsrc, int ns, int seis_form, int ishot, int sws);

void  outseis_glob(FILE *fp, FILE *fpdata, int comp, float **section,
int **recpos, int **recpos_loc, int ntr, float ** srcpos_loc,
int nsrc, int ns, int seis_form, int ishot, int sws);

void  outseis_vector(FILE *fp, FILE *fpdata, int comp, float *section,
int **recpos, int **recpos_loc, int ntr, float ** srcpos_loc,
int nsrc, int ns, int seis_form, int ishot, int sws);

void  inseis(int comp, float **section, int ntr, int ns, int sws, int iter);

void  taper(float **sectionpdiff, int ntr, int ns);

void  output_source_signal(FILE *fp, float **signals, int ns, int seis_form);

void PCG(float * PCG_new, float * PCG_old, float * PCG_dir, int PCG_class);

void PML_pro(float * d_x, float * K_x, float * alpha_prime_x, float * a_x, float * b_x, 
float * d_x_half, float * K_x_half, float * alpha_prime_x_half, float * a_x_half, float * b_x_half,
float * d_y, float * K_y, float * alpha_prime_y, float * a_y, float * b_y, 
float * d_y_half, float * K_y_half, float * alpha_prime_y_half, float * a_y_half, float * b_y_half);

void psource(int nt, float ** sxx, float ** syy,
float **  srcpos_loc, float ** signals, int nsrc, int sw);

float *rd_sour(int *nts,FILE* fp_source);

void read_density_glob(float ** rho_grav, int sws);

float readdsk(FILE *fp_in, int format);

void read_checkpoint(int nx1, int nx2, int ny1, int ny2,
float **  vx, float ** vy, float ** sxx, float ** syy, float ** sxy);

float **read_grav_pos(int *ngrav);

void read_par(FILE *fp_in);

void read_par_inv(FILE *fp,int nstage,int stagemax);

int **receiver(FILE *fp, int *ntr, int ishot);

void save_checkpoint(int nx1, int nx2, int ny1, int ny2,
float **  vx, float ** vy, float ** sxx, float ** syy, float ** sxy);

void saveseis(FILE *fp, float **sectionvx, float **sectionvy,float **sectionp,
float **sectioncurl, float **sectiondiv, int  **recpos, int  **recpos_loc, 
int ntr, float ** srcpos_loc, int nsrc,int ns, int iter);

void saveseis_glob(FILE *fp, float **sectionvx, float **sectionvy,float **sectionp,
float **sectioncurl, float **sectiondiv, int  **recpos, int  **recpos_loc, 
int ntr, float ** srcpos_loc, int nsrc,int ns, int iter);

void scale_grad(float ** A, float a, float ** B, int n, int m);

void snap(FILE *fp,int nt, int nsnap, float **vx, float **vy, float **sxx,
	float **syy, float **u, float **pi, float *hc);

void snapmerge(int nsnap);

float **sources(int *nsrc);

void solvelin(float  **AA, float *bb, float *x, int e, int method);

void seismo(int lsamp, int ntr, int **recpos, float **sectionvx, 
float **sectionvy, float **sectionp, float **sectioncurl, float **sectiondiv,
float **pvx, float **pvy, float **psxx, float **psyy, float **ppi, float **pu); 

void seismo_ssg(int lsamp, int ntr, int **recpos, float **sectionvx, 
float **sectionvy, float **sectionp, float **sectioncurl, float **sectiondiv,
float **pvx, float **pvy, float **psxx, float **psyy, float **ppi, float **pu,
float **prho, float *hc);

float **splitsrc(float **srcpos,int *nsrc_loc, int nsrc);

float **splitsrc_back(int **recpos,int *nsrc_loc, int nsrc);

void stalta(float **sectionp, int ntr, int nst, float *picked_times, int ishot);

void stf(float **sectionvy_obs, float **sectionvy, int ntr_glob, int ishot, int ns, int iter, int nshots, float **signals, int **recpos, float **srcpos);

void  timedomain_filt(float ** data, float fc, int order, int ntr, int ns, int method);
void  timedomain_filt_vector(float * data, float fc, int order, int ntr, int ns, int method);

void time_window(float **sectiondata, float * picked_times, int iter, int ntr_glob, int **recpos_loc, int ntr, int ns, int ishot);

void time_window_stf(float **sectiondata, int iter, int ntr_glob, int ns, int ishot);

void tripd(float *d, float *e, float *b, int n);

float ** wavelet(float ** srcpos_loc, int nsrc, int ishot);

float ** wavelet_stf(int nsrc, int ishot, float ** signals_stf);

void  wavelet_su(int comp, float **section, int ntr, int ns, int nsrc_loc, float ** srcpos_loc);

void  wavenumber(float ** grad);

void write_par(FILE *fp);

void writedsk(FILE *fp_out, float amp, int format);

void writemod(char modfile[STRING_SIZE], float ** array, int format);

void zero_LBFGS(int NLBFGS, int NLBFGS_vec, float * y_LBFGS, float * s_LBFGS, float * q_LBFGS, float * r_LBFGS, 
                 float * alpha_LBFGS, float * beta_LBFGS, float * rho_LBFGS);

void zero_PCG(float * PCG_old, float * PCG_new, float * PCG_dir, int PCG_vec);
		 
void FLnode(float  **  rho, float **  pi, float **  u, float **  taus, float **  taup, float *  eta);

void smooth_grad(float ** waveconv, float ** vel_mod);

void  smooth2(float ** grad);

/* declaration of functions for parser*/

/* declaration of functions for json parser in json_parser.c*/
int read_objects_from_intputfile(FILE *fp, char input_file[STRING_SIZE],char ** varname_list,char ** value_list);

void print_objectlist_screen(FILE *fp, int number_readobject,char ** varname_list,char ** value_list);

int count_occure_charinstring(char stringline[STRING_SIZE], char teststring[]);

void copy_str2str_uptochar(char string_in[STRING_SIZE], char string_out[STRING_SIZE], char teststring[]);

int get_int_from_objectlist(char string_in[STRING_SIZE], int number_readobject, int * int_buffer,
		char ** varname_list,char ** value_list);

int get_float_from_objectlist(char string_in[STRING_SIZE], int number_readobject, float * double_buffer,
		char ** varname_list,char ** value_list);

int get_string_from_objectlist(char string_in[STRING_SIZE], int number_readobject, char string_buffer[STRING_SIZE],
		char ** varname_list,char ** value_list);

int is_string_blankspace(char string_in[STRING_SIZE]);

void remove_blankspaces_around_string(char string_in[STRING_SIZE] );

void add_object_tolist(char string_name[STRING_SIZE],char string_value[STRING_SIZE], int * number_read_object,
		char ** varname_list,char ** value_list );

/* utility functions */
void err(char err_text[]);
void warning(char warn_text[]);
double maximum(float **a, int nx, int ny);
float *vector(int nl, int nh);
int *ivector(int nl, int nh);
double *dvector(int nl, int nh);
float **fmatrix(int nrl, int nrh, int ncl, int nch);
int *ivector(int nl, int nh);

float **matrix(int nrl, int nrh, int ncl, int nch);
int **imatrix(int nrl, int nrh, int ncl, int nch);
float ***f3tensor(int nrl, int nrh, int ncl, int nch,int ndl, int ndh);
void free_vector(float *v, int nl, int nh);
void free_dvector(double *v, int nl, int nh); 
void free_ivector(int *v, int nl, int nh);
void free_matrix(float **m, int nrl, int nrh, int ncl, int nch);
void free_imatrix(int **m, int nrl, int nrh, int ncl, int nch);
void free_f3tensor(float ***t, int nrl, int nrh, int ncl, int nch, int ndl, 
int ndh);
void zero(float *A, int u_max);
void normalize_data(float **data, int ntr, int ns);
