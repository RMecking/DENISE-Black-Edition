import sys,json,hashlib,os,argparse
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W))
parser=argparse.ArgumentParser(description='Explicitly authorized legacy reload fixture only.')
parser.add_argument('--evidence-root',type=Path,default=Path(os.environ.get('DENISE_PSV_EVIDENCE_ROOT',W/'benchmark-results/psv-multishot')))
parser.add_argument('--truth',type=Path,required=True)
parser.add_argument('--legacy-binary',type=Path,required=True)
args=parser.parse_args();E=args.evidence_root.resolve()
from tests.physics.test_psv_multishot_frequency import prepare
from tests.physics.test_visco_psv_physical_gradient_oracle import ViscoPSVOracleConfig,_reference_model,PHYSICAL_FIELDS
from tests.physics.test_m8e2c5_cuda_active_fwi import _set_parameters,_set_workflow
from tests.physics.test_visco_psv_active_fwi import _read_float_grid
from tests.utilities.runner import run_denise,result_summary
D=E/'b3s-r3-r4/legacy-stage-reload-check';D.mkdir(parents=True,exist_ok=True)
truth=args.truth.resolve();binary=args.legacy_binary.resolve();c=ViscoPSVOracleConfig();m=_reference_model(c)
os.environ.update(DENISE_PSV_BACKEND='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
for key in ['DENISE_PSV_EXACT_VISCO_GRADIENT','DENISE_PSV_EXACT_DISTRIBUTED_GRADIENT_ONLY']:os.environ.pop(key,None)
def run(name,model,stages):
 case=D/name;prepare(case,c,model,observed=truth,stages=stages,layout=(2,2))
 _set_parameters(case,ITERMAX=4,MIN_ITER=4,PCG_BETA=0,INV_MOD_OUT=1,EPS_SCALE='.01',STEPMAX=6)
 _set_workflow(case,PRO=0.,INV_RHO_ITER=999999,INV_QS_ITER=999999)
 # _set_workflow changes the first row; apply fixed fields to all rows.
 path=case/'workflow.inp';lines=path.read_text().splitlines();header=lines[0].split()
 for i in range(1,len(lines)):
  row=lines[i].split()
  for k,v in dict(PRO=0.,INV_RHO_ITER=999999,INV_QS_ITER=999999).items():row[header.index(k)]=str(v)
  lines[i]=' '.join(row)
 path.write_text('\n'.join(lines)+'\n')
 r=run_denise(repository_root=W,case_directory=case,denise_binary=binary,mpiexec='mpiexec',ranks=4,configuration={'gate':'B3-S-legacy-stage-reload-identity','optimizer':'PCG_BETA0','prospective':'four iterations each at12/18Hz, byte-identical stage/final models and misfit numeric rows; nonzeroVp/Vs changes required'},timeout_seconds=400)
 assert r.returncode==0,result_summary(r)
 return case
def field(case,stage,f):return case/'model'/f'accepted_{f}_stage_{stage}_it_4.bin'
# Fixture INV_MODELFILE comes from the normal _write_case default.
def find(case,stage,f):
 matches=list((case/'model').glob(f'*_{f}_stage_{stage}_it_4.bin'));assert len(matches)==1,matches;return matches[0]
combined=run('combined',m,(12.,18.));first=run('split-first',m,(12.,))
for f in ['vp','vs','rho']:assert find(combined,1,f).read_bytes()==find(first,1,f).read_bytes(),('stage1',f)
restart={f:(_read_float_grid(find(first,1,f),c.cell_count) if f in ['vp','vs','rho'] else m[f]) for f in PHYSICAL_FIELDS}
second=run('split-second',restart,(18.,))
checks={}
for f in ['vp','vs','rho']:
 a=find(combined,2,f);b=find(second,1,f);assert a.read_bytes()==b.read_bytes(),('stage2',f)
 checks[f]={'combined_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'split_sha256':hashlib.sha256(b.read_bytes()).hexdigest()}
for f in ['vp','vs']:assert list(_read_float_grid(find(combined,2,f),c.cell_count))!=list(m[f]),('vacuous unchanged model',f)
def history(case):
 return [[float(v) for v in line.split()[:-1]] for line in (case/'misfit.log').read_text().splitlines()]
assert history(combined)==history(first)+history(second)
report={'source':'b07d81d78aabdda54ebc6023fe99265174291753 + package inseis bounds fix','binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'layout':[2,2],'four_iterations_per_stage':True,'nonzero_vp_vs_change':True,'model_identity':checks,'misfit_rows_identical':True,'state_reasoning':'iter reset each stage; EPS_SCALE copied to local search each iteration; PCG_BETA0 removes prior direction dependence; rho fixed, Q forced30 on read; EPRECOND0, COMP_WEIGHT0, GRAVITY0 leave iter_true-dependent C averages unused by active equations'}
(D/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
