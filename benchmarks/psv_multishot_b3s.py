#!/usr/bin/env python3
"""Ratified B3-S (R3+R4), local bounded experiment, never a Scientific PASS.

Inputs are the unmodified extracted package and explicitly built production
binaries. Each command is a separate step; inspect estimates before inversion.
The full benchmark requires a separately authorized execution host.
"""
import argparse, hashlib, json, math, os, re, shutil, signal, struct, subprocess, time
from pathlib import Path
import numpy as np
TRAIN=(1,26,51,76,100); SHOTS=TRAIN+(38,)
W=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('DENISE_PSV_EVIDENCE_ROOT',W/'benchmark-results/psv-multishot')).resolve()
D=ROOT/'b3s-r3-r4'; O=D/'original/denise_fwi_testpaket_kiel/denise_legacy/par'
PACKAGE=ROOT/'benchmark-inputs/denise_fwi_testpaket_kiel/marmousi/start'
FILTER=D/'production-filter'; LEGACY=D/'koehn-b07d81d/bin/denise'
SOURCE='200b3b8c4cb7c3141e8e5bde0af1db17c8c3b590'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):(D/name).write_text(json.dumps(value,indent=2,allow_nan=False))
def names(left):
 ans={left,left.split('_(',1)[0]};m=re.search(r'\(([^()]*)\)\s*$',left)
 if m:ans.add(m[1]);ans.update(x.strip() for x in m[1].split(','))
 return ans
def replace_parameters(text,updates):
 lines=text.splitlines()
 for key,value in updates.items():
  hits=[i for i,l in enumerate(lines) if '=' in l and not l.lstrip().startswith('#') and key in names(l.rsplit('=',1)[0].strip())]
  if len(hits)!=1:raise ValueError((key,hits))
  i=hits[0];lines[i]=lines[i].rsplit('=',1)[0].rstrip()+' = '+str(value)
 return '\n'.join(lines)+'\n'
def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def tau_contract():
 # Actual unmodified production q_parameterization.o, not a fitted surrogate.
 result=json.loads(subprocess.check_output([str(D/'q-contract')],text=True))
 result.update(FL=3.,Q_APPROX_FMIN=1.,Q_APPROX_FMAX=10.,Q_APPROX_DF=.25,physical_q=80.)
 assert result['tau_f32']==result['legacy_tau_f32']
 return result

def filtered(a,order,fc,low=0.,label='filter'):
 a=np.ascontiguousarray(a,dtype='<f4');a=a.reshape(-1,3000)
 cache=D/'filtered';cache.mkdir(exist_ok=True)
 key=hashlib.sha256(a.tobytes()+repr((order,fc,low)).encode()).hexdigest();ip=cache/(key+'.input.f32');op=cache/(key+'.output.f32')
 if not op.exists():
  a.tofile(ip);subprocess.run([str(FILTER),'3000',str(len(a)),'.002',str(order),str(fc),str(low),str(ip),str(op),'exact-production-filter'],check=True,timeout=30)
 return np.fromfile(op,dtype='<f4').reshape(a.shape)
def prepare():
 tc=tau_contract();models=D/'models';models.mkdir(exist_ok=True)
 truevs=np.fromfile(PACKAGE/'marmousi_II_marine.vs',dtype='<f4').reshape(500,174)
 removed=np.where(np.any(truevs==0,axis=0))[0];keep=np.where(np.all(truevs!=0,axis=0))[0]
 assert list(removed)==list(range(22)) and len(keep)==152
 bindings={}
 for role in ['truth','start']:
  out=models/role;out.mkdir(exist_ok=True)
  for field in ['vp','vs','rho']:
   prefix='marmousi_II_marine' if role=='truth' else ('marmousi_II_smooth2' if field=='rho' else 'marmousi_II_start_1D')
   inp=PACKAGE/(prefix+'.'+field);a=np.fromfile(inp,dtype='<f4').reshape(500,174)[:,keep].copy()
   assert np.isfinite(a).all() and np.min(a)>0
   p=out/('model.'+field);a.tofile(p);bindings[str(p)]={'input':str(inp),'input_sha256':sha(inp),'sha256':sha(p),'min':float(a.min()),'max':float(a.max())}
  for field in ['qp','qs']:
   p=out/('model.'+field);np.full((500,152),80.,dtype='<f4').tofile(p);bindings[str(p)]={'sha256':sha(p)}
 impulse=np.zeros((1,3000),dtype='<f4');impulse[0,0]=1.
 shaped=filtered(impulse,5,15.,-5.,'spike')
 for fc in [2.,3.5]:
  source=filtered(shaped,6,fc)
  path=D/f'stage-source-{fc}.dat';path.write_text(''.join(f'{float(v):.9e}\n' for v in source[0]))
  assert np.array([float(l) for l in path.read_text().splitlines()],dtype='<f4').tobytes()==source.tobytes()
 save('preparation.json',dict(source=SOURCE,tau_contract=tc,nx=500,ny=152,dh=20.,nt=3000,dt=.002,time=6.,removed_top_rows_0based=removed.tolist(),removed_bottom_rows=0,source_row=1,source_y=20.,receiver_row=2,receiver_y=40.,cpml_width=10,acquisition_inside_upper_cpml=True,original_source_depth_equivalent=460.,original_receiver_depth_equivalent=480.,model_bindings=bindings,training=TRAIN,control=38,filter_binary_sha256=sha(FILTER),filter_sources={str(p):sha(p) for p in [W/'src/apply_tdfilt.c',W/'src/timedomain_filt.c',W/'benchmarks/psv_multishot_production_filter.c']},source_ascii_roundtrip=True))
 print('prepared B3-S 500x152, Q/tau',tc,flush=True)
def write_case(name,backend,mode,*,model=None,shots=SHOTS,stage_source=None,stages=(2.,3.5),iterations=4,observed=False):
 case=D/name
 if case.exists():raise FileExistsError(case)
 case.mkdir()
 for folder in ['su','log','model','jacobian','snap','start','taper','trace_kill','picked_times','gravity','wavelet']: (case/folder).mkdir()
 tc=tau_contract();model=model or D/'models/truth'
 for field in ['vp','vs','rho','qp','qs']:
  dest=case/'start'/('model.'+field)
  if backend=='koehn' and field in ['qp','qs']:np.full((500,152),tc['legacy_q_f32'],dtype='<f4').tofile(dest)
  else:shutil.copyfile(model/('model.'+field),dest)
 raw=(O/'source/source_OBC_VSP.dat').read_text().splitlines()[1:];rows=[]
 for shot in shots:
  values=raw[shot-1].split();values[2]='20.0';rows.append(' '.join(values))
 (case/'source.dat').write_text(str(len(shots))+'\n'+'\n'.join(rows)+'\n')
 recs=[line.split() for line in (O/'receiver/receiver_OBC.dat').read_text().splitlines() if line.strip()]
 assert len(recs)==400
 (case/'receivers.dat').write_text(''.join(f'{r[0]} 40.0\n' for r in recs))
 layout=(1,1) if backend=='cuda' else (4,2)
 updates=dict(MODE=mode,PHYSICS=1,NPROCX=layout[0],NPROCY=layout[1],FD_ORDER=4,NX=500,NY=152,DH=20.,TIME=6.,DT=.002,QUELLART=3 if stage_source else 6,SOURCE_FILE='source.dat',REC_FILE='receivers',MFILE='start/model',L=1,FL=tc['FL'],FREE_SURF=0,FW=10,NDT=1,DTINV=1,GRAD_FORM=2,INVMAT1=1,GRAD_METHOD=1,PCG_BETA=0 if backend=='koehn' else 1,ITERMAX=iterations,MIN_ITER=iterations,INV_MOD_OUT=1,INV_MODELFILE='model/inverted',JACOBIAN='jacobian/gradient',MISFIT_LOG_FILE='misfit.log',SEIS_FILE_VX='su/synthetic_x.su',SEIS_FILE_VY='su/synthetic_y.su',DATA_DIR='observed',TESTSHOT_START=f'1, {len(shots)}, 1',SWS_TAPER_GRAD_HOR=0,SWS_TAPER_GRAD_VERT=0,SWS_TAPER_GRAD_SOURCES=0,SWS_TAPER_CIRCULAR_PER_SHOT=0,SWS_TAPER_FILE=0,MODEL_FILTER=0,GRAD_FILTER=0,LOG=1,WRITE_STF=1,WRITEMOD=1,STEPMAX=6)
 if stage_source:
  updates['SIGNAL_FILE']='wavelet/external'
  for k in range(1,len(shots)+1):shutil.copyfile(stage_source,case/f'wavelet/external_shot_{k}.dat')
 text=replace_parameters((O/'DENISE_marm_OBC.inp').read_text(),updates)
 if backend!='koehn':text+='Q_PARAMETERIZATION_MODE =1\nQ_APPROX_FMIN =1.0\nQ_APPROX_FMAX =10.0\nQ_APPROX_DF =0.25\n'
 (case/'denise.inp').write_text(text)
 flow=(O/'FWI_workflow_marmousi.inp').read_text().splitlines();header=flow[0].split();template=flow[1].split();out=[]
 for fc in stages:
  values=template.copy()
  for k,v in dict(PRO=0.,TIME_FILT=1,FC_low=0.,FC_high=fc,ORDER=6,INV_VP_ITER=0,INV_VS_ITER=0,INV_RHO_ITER=999999,INV_QS_ITER=999999,EPRECOND=0,NORMALIZE=0,N_ORDER=0).items():values[header.index(k)]=str(v)
  out.append(' '.join(values))
 (case/'workflow.inp').write_text(flow[0]+'\n'+'\n'.join(out)+'\n')
 if observed:
  for k,shot in enumerate(shots,1):
   for c in ['x','y']:shutil.copyfile(D/'bank/su'/f'synthetic_{c}.su.shot{SHOTS.index(shot)+1}',case/f'observed_{c}.su.shot{k}')
 return case,layout

def descendants(pid):
 found={pid};pending=[pid]
 while pending:
  p=pending.pop()
  try:children=[int(v) for v in Path(f'/proc/{p}/task/{p}/children').read_text().split()]
  except OSError:continue
  for child in children:
   if child not in found:found.add(child);pending.append(child)
 return found

def run_case(case,backend,layout,*,gradient_only=False,timeout=1790.):
 binary=LEGACY if backend=='koehn' else W/'bin'/('denise_cuda' if backend=='cuda' else 'denise')
 cmd=['mpiexec','--oversubscribe','-np',str(layout[0]*layout[1]),str(binary),'denise.inp','workflow.inp']
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DENISE_PSV_BACKEND='cuda' if backend=='cuda' else 'cpu',DENISE_PSV_EXACT_SEGMENTS='8',GIT_CEILING_DIRECTORIES=str(W.parent))
 for key in ['DENISE_PSV_EXACT_VISCO_GRADIENT','DENISE_PSV_EXACT_DISTRIBUTED_GRADIENT_ONLY','DENISE_CUDA_PSV_TEST_USE_RAW_MODELED_OBSERVED','DENISE_PSV_DIAGNOSTIC_CPU_RESIDUAL']:env.pop(key,None)
 if gradient_only:env['DENISE_PSV_EXACT_DISTRIBUTED_GRADIENT_ONLY']='1'
 input_files=[case/'denise.inp',case/'workflow.inp',case/'source.dat',case/'receivers.dat',*sorted((case/'start').glob('model.*')),*sorted(case.glob('observed*.su.shot*')),*sorted((case/'wavelet').glob('external*'))]
 info=dict(source=('b07d81d78aabdda54ebc6023fe99265174291753 + package-prescribed inseis bounds fix' if backend=='koehn' else SOURCE),harness_sha256=sha(__file__),binary=str(binary),binary_sha256=sha(binary),command=cmd,input_sha256={str(p.relative_to(case)):sha(p) for p in input_files},backend=backend,layout=layout,segments=8,gradient_only=gradient_only)
 started=time.monotonic();peak=0;gpu_peak=None;gpu_baseline=None;last_gpu=0.;reason=None
 with (case/'stdout.log').open('w') as out,(case/'stderr.log').open('w') as err:
  process=subprocess.Popen(cmd,cwd=case,env=env,stdout=out,stderr=err,start_new_session=True)
  while process.poll() is None:
   rss=0
   for pid in descendants(process.pid):
    try:rss+=int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    except (OSError,IndexError):pass
   peak=max(peak,rss)
   if backend=='cuda' and time.monotonic()-last_gpu>1.:
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=5)
    try:
     memory=int(r.stdout.strip().splitlines()[0]);gpu_peak=max(gpu_peak or 0,memory);gpu_baseline=memory if gpu_baseline is None else gpu_baseline
    except (ValueError,IndexError):pass
    last_gpu=time.monotonic()
   if rss>=8*1024**3:reason='8GiB aggregate descendant RSS ceiling'
   if time.monotonic()-started>=timeout:reason='30min wall ceiling'
   if reason:os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10);break
   time.sleep(.1)
 info.update(returncode=process.returncode,runtime_seconds=time.monotonic()-started,peak_aggregate_rss_bytes=peak,gpu_device_total_peak_mib=gpu_peak,gpu_first_observed_mib=gpu_baseline,gpu_memory_note='Whole-device nvidia-smi total, includes desktop/background; isolated solver GPU job',stop_reason=reason)
 (case/'run.json').write_text(json.dumps(info,indent=2));print(case.name,info['returncode'],info['runtime_seconds'],'s',peak/1024**2,'MiB',flush=True)
 if reason or process.returncode:raise RuntimeError('Run did not complete: '+str(case)+' '+str(info))
 return info

def su(path):
 raw=Path(path).read_bytes();ns=struct.unpack_from('<H',raw,114)[0];assert ns==3000
 rec=240+4*ns;assert len(raw)%rec==0
 return np.frombuffer(raw,dtype=np.uint8).reshape(-1,rec)[:,240:].copy().view('<f4').reshape(-1,ns)
def forwardcheck():
 result={}
 for backend in ['cpu','koehn']:
  case,layout=write_case('forwardcheck-'+backend,backend,0,shots=(1,));run_case(case,backend,layout)
  result[backend]={c:su(case/'su'/f'synthetic_{c}.su.shot1') for c in ['x','y']}
 compare={}
 for c in ['x','y']:
  a=result['cpu'][c].astype(float);b=result['koehn'][c].astype(float)
  assert np.isfinite(a).all() and np.isfinite(b).all() and np.linalg.norm(a)>0
  compare[c]={'relative_l2':float(np.linalg.norm(a-b)/np.linalg.norm(a)),'max_abs_difference':float(np.max(np.abs(a-b))),'byte_equal':a.tobytes()==b.tobytes()}
 save('forwardcheck.json',dict(configuration='Unstaged package spike; same source type/amplitude/shaping. Legacy b07 reader overwrites both Q with30; transformed input is ignored. No tau-equivalence or isolated optimizer claim.',comparison=compare,q_mismatch=dict(candidate_tau=tau_contract()['tau_f32'],legacy_tau=f32(2/30),legacy_forced_q=30,legacy_input_q_ignored=tau_contract()['legacy_q_f32'],source='b07d81d src/PSV/readmod_visc_PSV.c qp=30;qs=30')));print(compare,flush=True)
def bank():
 assert (D/'forwardcheck.json').exists()
 case,layout=write_case('bank','cpu',0);run_case(case,'cpu',layout)
 for k in range(1,7):
  for c in ['x','y']:assert np.isfinite(su(case/'su'/f'synthetic_{c}.su.shot{k}')).all()
 save('bank.json',{str(p.relative_to(case)):sha(p) for p in sorted((case/'su').glob('synthetic*'))})
def pilot():
 assert (D/'bank.json').exists();result={}
 for backend in ['cpu','cuda']:
  case,layout=write_case('pilot-'+backend,backend,1,model=D/'models/start',shots=(1,),stages=(2.,),iterations=1,observed=True)
  result[backend]=run_case(case,backend,layout,gradient_only=True)
 # Conservative 8 iterations x5 shots x(gradient +6 trial forwards),1.5 margin.
 f=json.loads((D/'forwardcheck-cpu/run.json').read_text())['runtime_seconds']
 estimates={b:8*5*(v['runtime_seconds']+6*f)*1.5 for b,v in result.items()}
 estimates['koehn']=8*5*10*f*1.5
 save('runtime-estimate.json',dict(pilots=result,cpu_single_shot_forward_seconds=f,conservative_estimate_seconds=estimates,formula='8 iterations*5 shots*(single-shot gradient +6 complete forwards)*1.5; legacy10 forward equivalents',within_30min=all(v<1800 for v in estimates.values())))
 print('prospective main runtime estimate',estimates,flush=True)
def refinepilot():
 # A whole iteration amortizes MPI/model-allocation setup exactly as main.
 # Existing grad-only estimate is preserved, never relabeled as a bound.
 result={}
 for backend in ['cpu','cuda','koehn']:
  case,layout=write_case('iteration-pilot-'+backend,backend,1,model=D/'models/start',shots=TRAIN,stages=(2.,),iterations=1,observed=True)
  result[backend]=run_case(case,backend,layout)
  save('iteration-pilots.json',result)
  rows=(case/'misfit.log').read_text().splitlines();assert len(rows)==1,(backend,rows)
 print('Full-iteration pilots complete; derive prospective bounds before main.',flush=True)

def forwardtiming():
 result={};pipeline={}
 for backend in ['cpu','cuda','koehn']:
  result[backend]={};pipeline[backend]={}
  for fc in [2.,3.5]:
   case=D/f'runtime-forward-{backend}-fc{fc}'
   if case.exists():
    info=json.loads((case/'run.json').read_text());assert info['returncode']==0
    assert sha(info['binary'])==info['binary_sha256']
    assert all(sha(case/rel)==value for rel,value in info['input_sha256'].items())
    result[backend][str(fc)]=info
   else:
    case,layout=write_case(case.name,backend,0,model=D/'models/start',stage_source=D/f'stage-source-{fc}.dat')
    result[backend][str(fc)]=run_case(case,backend,layout)
   if fc==2.:
    # Compare exact finite-trace source handoff to authoritative production
    # reversed residuals, written only for original/local shot1 at iter1.
    agreement={};total=0.
    for c in ['x','y']:
     residual=(su(case/'su'/f'synthetic_{c}.su.shot1')-filtered(su(D/'bank/su'/f'synthetic_{c}.su.shot1'),6,fc)).astype('<f4')
     production=su(D/f'iteration-pilot-{backend}/su'/f'synthetic_{c}.su.shot1.it1')[:,::-1]
     agreement[c]=residual[:,1:-1].tobytes()==production[:,1:-1].tobytes()
     assert agreement[c],(backend,c,'retained production/evaluation residual mismatch')
    for k in range(1,6):
     for c in ['x','y']:
      r=(su(case/'su'/f'synthetic_{c}.su.shot{k}')-filtered(su(D/'bank/su'/f'synthetic_{c}.su.shot{k}'),6,fc)).astype('<f4').astype(float)
      total+=.5*float(np.sum(r[:,1:-1]**2,dtype=np.float64))
    pipeline[backend]=dict(retained_residuals_byte_equal=agreement,external_training_r4_objective=total)
    if backend!='koehn':
     prod=json.loads((D/f'iteration-pilot-{backend}/jacobian/gradient.active_fwi.json').read_text())['base_objective']
     pipeline[backend].update(production_training_r4_objective=prod,relative_accumulation_difference=abs(total-prod)/prod)
   save('forward-batch-timings.json',result);save('pipeline-check.json',pipeline)
 print('Forward-batch timings and direct residual equivalence recorded.',flush=True)

def estimate():
 pilots=json.loads((D/'iteration-pilots.json').read_text());forward=json.loads((D/'forward-batch-timings.json').read_text());out={}
 for backend in ['cpu','cuda','koehn']:
  f=max(v['runtime_seconds'] for v in forward[backend].values())
  if backend=='koehn':
   observed=len(re.findall(r'Starting simulation \(test-forward model\)',(D/'iteration-pilot-koehn/stdout.log').read_text()))//5
   maximum=14 # <=7 initial shrink trials, <=7 secondary bracket trials.
  else:
   observed=json.loads((D/f'iteration-pilot-{backend}/jacobian/gradient.active_fwi.json').read_text())['trial_forwards_max'];maximum=7
  assert 0<=observed<=maximum
  iteration=pilots[backend]['runtime_seconds']+(maximum-observed)*f
  combined=8*iteration*1.25;per_run=combined/2 if backend=='koehn' else combined
  out[backend]=dict(observed_pilot_trials=observed,maximum_trial_batches=maximum,six_shot_forward_batch_upper_seconds=f,measured_iteration_seconds=pilots[backend]['runtime_seconds'],estimated_complete_iteration_with_max_trials_seconds=iteration,reserve_factor=1.25,combined_estimate_seconds=combined,per_run_estimate_seconds=per_run,execution='two natural 4-iteration frequency stages with proven reload identity' if backend=='koehn' else 'one two-stage run',within_30min=per_run<1800,peak_pilot_rss_bytes=pilots[backend]['peak_aggregate_rss_bytes'])
  assert out[backend]['within_30min'],('STOP prospective over-limit run',backend,out[backend])
 assert (D/'legacy-stage-reload-check/result.json').exists()
 save('runtime-estimate-refined.json',dict(method='Full iteration includes all setup; missing worst-case trial batches use slower measured six-shot stage forward (conservative for five training shots);25% reserve; frozen original coarse estimate preserved',backends=out));print(json.dumps(out,indent=2),flush=True)

def invert():
 estimates=json.loads((D/'runtime-estimate-refined.json').read_text())['backends'];result={}
 for backend in ['cpu','cuda']:
  assert estimates[backend]['within_30min']
  case,layout=write_case('invert-'+backend,backend,1,model=D/'models/start',shots=TRAIN,observed=True)
  result[backend]=run_case(case,backend,layout)
  rows=(case/'misfit.log').read_text().splitlines();assert len(rows)==8,(backend,rows)
  assert [int(l.split()[-1]) for l in rows]==[1]*4+[2]*4
  for field in ['rho','qp','qs']:
   for stage in [1,2]:
    p=persisted_field(case,backend,stage,field);assert sha(p)==sha(D/'models/start'/('model.'+field)),(backend,'fixed '+field)
  save('inversions.json',result)

def invertlegacy():
 estimate=json.loads((D/'runtime-estimate-refined.json').read_text())['backends']['koehn'];assert estimate['within_30min']
 proof=json.loads((D/'legacy-stage-reload-check/result.json').read_text());assert proof['misfit_rows_identical'] and proof['nonzero_vp_vs_change']
 result=[];model=D/'models/start';restart_binding={}
 for stage,fc in [(1,2.),(2,3.5)]:
  case,layout=write_case(f'invert-koehn-stage{stage}','koehn',1,model=model,shots=TRAIN,stages=(fc,),observed=True)
  result.append(run_case(case,'koehn',layout))
  rows=(case/'misfit.log').read_text().splitlines();assert len(rows)==4,(stage,rows)
  assert all(int(l.split()[-1])==1 for l in rows)
  rho=persisted_field(case,'koehn',stage,'rho');assert sha(rho)==sha(D/'models/start/model.rho')
  if stage==1:
   model=D/'legacy-stage2-input';model.mkdir()
   for field in ['vp','vs','rho']:
    inp=persisted_field(case,'koehn',1,field);out=model/('model.'+field);shutil.copyfile(inp,out);restart_binding[field]=dict(stage1_persisted_sha256=sha(inp),stage2_input_sha256=sha(out));assert sha(inp)==sha(out)
   for field in ['qp','qs']:shutil.copyfile(D/'models/start'/('model.'+field),model/('model.'+field))
  save('legacy-inversions.json',dict(natural_stage_reload=True,stages=result,reload_binding=restart_binding,proof='legacy-stage-reload-check/result.json',q_effective=30,pcg_beta=0,internal_j_is_legacy_not_common_R4=True))

def persisted_field(case,backend,stage,field):
 if backend=='koehn':return D/f'invert-koehn-stage{stage}'/'model'/f'inverted_{field}_stage_1_it_4.bin'
 return case/'model'/f'inverted_stage_{stage}_it_4.{field}'

def model_for(backend,stage):
 if stage==0:return D/'models/start'
 dest=D/'evaluated-models'/f'{backend}-stage{stage}';dest.mkdir(parents=True,exist_ok=True)
 case=D/('invert-'+backend)
 for field in ['vp','vs','rho']:
  inp=persisted_field(case,backend,stage,field)
  shutil.copyfile(inp,dest/('model.'+field))
 for field in ['qp','qs']:shutil.copyfile(D/'models/start'/('model.'+field),dest/('model.'+field))
 return dest

def evaluate():
 result={};metrics={}
 for backend in ['cpu','cuda','koehn']:
  result[backend]={};metrics[backend]={}
  for stage,fc in [(0,2.),(1,2.),(1,3.5),(2,3.5)]:
   model=model_for(backend,stage);key=f'modelstage{stage}-fc{fc}'
   case,layout=write_case('eval-'+backend+'-'+key,backend,0,model=model,stage_source=D/f'stage-source-{fc}.dat')
   info=run_case(case,backend,layout);per_shot=[];end_terms=[]
   for k,shot in enumerate(SHOTS,1):
    value=0.;last=0.
    for c in ['x','y']:
     synthetic=su(case/'su'/f'synthetic_{c}.su.shot{k}')
     observed=filtered(su(D/'bank/su'/f'synthetic_{c}.su.shot{k}'),6,fc)
     # The production residual is FP32 subtraction. Accumulate its square in FP64.
     residual=(synthetic-observed).astype('<f4').astype(float)
     assert np.isfinite(residual).all()
     value+=.5*float(np.sum(residual[:,1:-1]**2,dtype=np.float64));last+=.5*float(np.sum(residual[:,-1]**2,dtype=np.float64))
    per_shot.append(value);end_terms.append(last)
   result[backend][key]=dict(train=sum(per_shot[:5]),control=per_shot[5],per_shot=dict(zip(SHOTS,per_shot)),excluded_last_sample_term=dict(zip(SHOTS,end_terms)),runtime=info)
   save('external-misfits.json',result)
  for stage in [0,1,2]:
   model=model_for(backend,stage);fields={}
   for field in ['vp','vs']:
    a=np.fromfile(model/('model.'+field),dtype='<f4').reshape(500,152).astype(float);truth=np.fromfile(D/'models/truth'/('model.'+field),dtype='<f4').reshape(500,152).astype(float)
    bands=[np.arange(152),*np.array_split(np.arange(152),3)];labels=['total','upper','middle','lower']
    fields[field]={label:{'relative_l2':float(np.linalg.norm((a-truth)[:,idx])/np.linalg.norm(truth[:,idx])),'rmse_m_s':float(np.sqrt(np.mean((a-truth)[:,idx]**2))),'row_range_1based':[int(idx[0])+1,int(idx[-1])+1]} for label,idx in zip(labels,bands)}
   metrics[backend][stage]=fields
  save('model-errors.json',metrics)
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','forwardcheck','bank','pilot','refinepilot','forwardtiming','estimate','invert','invertlegacy','evaluate']);a=ap.parse_args();globals()[a.stage]()
