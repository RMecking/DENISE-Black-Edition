"""PSV-MULTISHOT-01: sequential shot sums and source/observed filtering.

Frozen criteria are imported from M7c and M8e-2C5 unchanged.
The central difference uses scalar production objectives, never the adjoint.
"""
from __future__ import annotations
import json
import math
from array import array
from pathlib import Path
import pytest
from tests.physics.test_visco_psv_physical_gradient_oracle import (
    EPSILONS, GREEN_RELATIVE_ERROR_MAX, PHYSICAL_FIELDS, ViscoPSVOracleConfig,
    _directions, _perturb, _reference_model, _target_model, _write_case,
    _require_future_physical_products,
)
from tests.physics.test_m8e2c5_cuda_active_fwi import (
    _configure_cuda_envelope, _set_parameters, _set_workflow, _relative_l2, cuda_denise_binary,
)
from tests.physics.test_visco_psv_distributed_gradient import _read_floats
from tests.utilities.runner import run_denise, result_summary
from tests.utilities.fwi_gradient import read_su_float_samples

pytestmark = pytest.mark.integration
SOURCES = ((180.0,170.0),(240.0,220.0))

def prepare(directory, config, model, *, observed=None, shots=(0,1),
            mode=1, filtered=True, stages=(18.0,), layout=(1,1)):
    _write_case(directory,config=config,model=model,mode=mode,
                nprocx=layout[0],nprocy=layout[1])
    _configure_cuda_envelope(directory,1)
    (directory/'source.dat').write_text(str(len(shots))+'\n'+''.join(
        f'{SOURCES[s][0]} 0.0 {SOURCES[s][1]} 0.0 13.0 1.0 20.0 1\n'
        for s in shots),encoding='ascii')
    if observed is not None:
        for local,s in enumerate(shots,1):
            for c in ('x','y'):
                (directory/f'observed_{c}.su.shot{local}').write_bytes(
                    (observed/'su'/f'synthetic_{c}.su.shot{s+1}').read_bytes())
    _set_workflow(directory,TIME_FILT=int(filtered),FC_low=0.0,FC_high=stages[0])
    flow=(directory/'workflow.inp').read_text().splitlines()
    names=flow[0].split()
    rows=[]
    for frequency in stages:
        values=flow[1].split(); values[names.index('FC_high')]=str(frequency)
        rows.append(' '.join(values))
    (directory/'workflow.inp').write_text(flow[0]+'\n'+'\n'.join(rows)+'\n')
    # Deliberately leave TESTSHOT_END=1: exact trials must still use every shot.

def run(directory, repository_root, binary, mpiexec, *, backend='cpu',
        gradient_only=True, layout=(1,1)):
    with pytest.MonkeyPatch.context() as env:
        env.setenv('DENISE_PSV_BACKEND',backend)
        env.setenv('DENISE_PSV_EXACT_VISCO_GRADIENT','1')
        env.setenv('DENISE_PSV_EXACT_SEGMENTS','4')
        env.setenv('DENISE_CUDA_PSV_EXACT_FWI_REPORT_FILE',str(directory/'cuda.json'))
        if backend=='cuda':
            env.setenv('DENISE_CUDA_PSV_PRODUCTION_RESIDUAL_FILE',str(directory/'uploaded_residual.f32'))
        if gradient_only:
            env.setenv('DENISE_PSV_EXACT_DISTRIBUTED_GRADIENT_ONLY','1')
        else:
            env.delenv('DENISE_PSV_EXACT_DISTRIBUTED_GRADIENT_ONLY',raising=False)
        result=run_denise(repository_root=repository_root,case_directory=directory,
            denise_binary=binary,mpiexec=mpiexec,ranks=layout[0]*layout[1],
            configuration={'gate':'PSV-MULTISHOT-01','backend':backend},
            timeout_seconds=240.0)
    assert result.returncode==0,result_summary(result)
    gradients={f:_read_floats(directory/'jacobian'/f'gradient.raw.{f}',
               ViscoPSVOracleConfig().cell_count) for f in PHYSICAL_FIELDS}
    if backend=='cuda':
        report=json.loads((directory/'cuda.json').read_text()); objective=report['base_objective']
    else:
        name='gradient.segmented_gradient.json' if layout==(1,1) else 'gradient.segmented_gradient.rank0.json'
        report=json.loads((directory/'jacobian'/name).read_text()); objective=report['global_objective']
    return {'directory':directory,'objective':objective,'gradients':gradients,'report':report}

@pytest.fixture(scope='module')
def experiment(tmp_path_factory,repository_root,denise_binary,mpiexec):
    root=tmp_path_factory.mktemp('psv_multishot_frequency')
    config=ViscoPSVOracleConfig(); model=_reference_model(config)
    truth=root/'truth'; prepare(truth,config,_target_model(model,config),mode=0,filtered=False)
    with pytest.MonkeyPatch.context() as env:
        env.setenv('DENISE_PSV_BACKEND','cpu')
        r=run_denise(repository_root=repository_root,case_directory=truth,
            denise_binary=denise_binary,mpiexec=mpiexec,ranks=1,
            configuration={'role':'truth','gate':'PSV-MULTISHOT-01'},timeout_seconds=120)
    assert r.returncode==0,result_summary(r)
    def calculate(name, candidate=model, shots=(0,1), layout=(1,1),filtered=True):
        d=root/name; prepare(d,config,candidate,observed=truth,shots=shots,layout=layout,filtered=filtered)
        return run(d,repository_root,denise_binary,mpiexec,layout=layout)
    multi=calculate('multi')
    singles=[calculate(f'single{s}',shots=(s,)) for s in range(2)]
    unfiltered=calculate('unfiltered',filtered=False)
    return dict(root=root,config=config,model=model,truth=truth,multi=multi,
                repository_root=repository_root,denise_binary=denise_binary,mpiexec=mpiexec,
                singles=singles,unfiltered=unfiltered,calculate=calculate)

def test_multishot_equals_ascending_single_shot_sum(experiment):
    e=experiment; expected=sum(s['objective'] for s in e['singles'])
    # R4 projection: reversed residual sample 1 is physical NT.
    for single in e['singles']:
        for component in ('x','y'):
            residual=read_su_float_samples(single['directory']/'su'/
                f'synthetic_{component}.su.shot1.it1',e['config'].receiver_count,
                e['config'].samples_per_trace)
            assert all(value==0.0 for value in residual[::e['config'].samples_per_trace])
            assert any(value!=0.0 for value in residual)
    assert math.isclose(e['multi']['objective'],expected,rel_tol=1e-14,abs_tol=0.0)
    for f in PHYSICAL_FIELDS:
        expected=array('f',(a+b for a,b in zip(*(s['gradients'][f] for s in e['singles']))))
        assert list(e['multi']['gradients'][f])==list(expected),f
    assert abs(e['multi']['objective']-e['unfiltered']['objective'])>0.01*e['unfiltered']['objective']

def test_filtered_multishot_all_five_central_differences(experiment):
    e=experiment; measurements={}; products={}; references={}
    for f,direction in _directions(e['config']).items():
        analytic=math.fsum(g*m*w for field,weights in direction.items()
            for g,m,w in zip(e['multi']['gradients'][field],e['model'][field],weights))
        rows=[]
        for eps in EPSILONS:
            plus=e['calculate'](f'{f}_{eps}_plus',_perturb(e['model'],direction,eps))
            minus=e['calculate'](f'{f}_{eps}_minus',_perturb(e['model'],direction,-eps))
            rows.append({'epsilon':eps,'derivative':(plus['objective']-minus['objective'])/(2*eps)})
        fd=math.fsum(row['derivative'] for row in rows[-2:])/2
        assert math.isfinite(fd)
        assert abs(fd)>1e-14,f
        assert math.isfinite(analytic)
        assert analytic*fd>0.0,f
        products[f]=analytic; references[f]={'derivative':fd}
        error=abs(analytic-fd)/max(abs(analytic),abs(fd),1e-30)
        spread=abs(rows[-1]['derivative']-rows[-2]['derivative'])/max(
            abs(rows[-1]['derivative']),abs(rows[-2]['derivative']),1e-30)
        measurements[f]=dict(analytic=analytic,central_difference=fd,relative_error=error,
                             window_spread=spread,rows=rows)
        assert spread<=GREEN_RELATIVE_ERROR_MAX,(f,measurements[f])
        assert error<=GREEN_RELATIVE_ERROR_MAX,(f,measurements[f])
    _require_future_physical_products(references,products)
    assert not math.isclose(references['qp']['derivative'],references['qs']['derivative'],rel_tol=.02,abs_tol=1e-14)
    (e['root']/'filtered_fd.json').write_text(json.dumps(measurements,indent=2))

def test_filtered_multishot_spatial_mpi_invariant(experiment):
    e=experiment; report={}
    for layout in ((2,1),(1,2),(2,2)):
        candidate=e['calculate'](f'mpi_{layout[0]}_{layout[1]}',layout=layout)
        # Same M8d byte-identical physical gradient claim, all five fields.
        for f in PHYSICAL_FIELDS:
            assert candidate['gradients'][f].tobytes()==e['multi']['gradients'][f].tobytes(),(layout,f)
        assert math.isclose(candidate['objective'],e['multi']['objective'],rel_tol=1e-14,abs_tol=0.0)
        report[str(layout)]=candidate['objective']
    (e['root']/'mpi.json').write_text(json.dumps(report,indent=2))

@pytest.mark.optional_prerequisite
def test_filtered_multishot_cuda_agreement(
        experiment,repository_root,denise_binary,cuda_denise_binary,mpiexec):
    e=experiment; config=e['config']; report={}
    gpu=e['root']/'gpu_gradient'
    prepare(gpu,config,e['model'],observed=e['truth'])
    candidate=run(gpu,repository_root,cuda_denise_binary,mpiexec,backend='cuda')
    uploaded=array('f'); uploaded.frombytes((gpu/'uploaded_residual.f32').read_bytes())
    # Chronological upload, both components: physical NT must be zero.
    assert all(value==0.0 for value in uploaded[config.samples_per_trace-1::config.samples_per_trace])
    for f in PHYSICAL_FIELDS:
        rel=_relative_l2(candidate['gradients'][f],e['multi']['gradients'][f])
        association=(candidate['report']['fields'][f]['expected_sum_fnv1a64']==
                     candidate['report']['fields'][f]['production_matrix_fnv1a64'])
        report[f]=dict(relative_l2=rel,association_matches=association)
    (e['root']/'cuda_agreement.json').write_text(json.dumps(report,indent=2))
    failures={f:entry for f,entry in report.items()
              if not(entry['relative_l2']<=2e-6 and entry['association_matches'])}
    assert not failures,failures

@pytest.mark.parametrize('backend',[
    'cpu',pytest.param('cuda',marks=pytest.mark.optional_prerequisite)])
def test_filtered_multishot_two_stage_reload(
        experiment,repository_root,denise_binary,mpiexec,backend,request):
    e=experiment; config=e['config']; report={}
    binary=denise_binary if backend=='cpu' else request.getfixturevalue('cuda_denise_binary')
    for backend,binary in ((backend,binary),):
        active=e['root']/f'{backend}_two_stage'
        prepare(active,config,e['model'],observed=e['truth'],stages=(12.0,18.0))
        result=run(active,repository_root,binary,mpiexec,backend=backend,gradient_only=False)
        history=[]
        for line in (active/'misfit.log').read_text().splitlines():
            values=line.split(); history.append((int(values[-1]),float(values[4]),float(values[7])))
        assert [row[0] for row in history]==[1,2]
        assert all(accepted<base for _,base,accepted in history)
        # The selected fixture INV_MODELFILE is read from the lifecycle record.
        lifecycle=json.loads((active/'jacobian'/'gradient.active_fwi.json').read_text())
        persisted=active/lifecycle['persisted_prefix']
        accepted={f:_read_floats(Path(str(persisted)+'.'+f),config.cell_count) for f in PHYSICAL_FIELDS}
        reload=e['root']/f'{backend}_reload'
        prepare(reload,config,accepted,observed=e['truth'],stages=(18.0,))
        reload_result=run(reload,repository_root,denise_binary,mpiexec)
        assert math.isclose(reload_result['objective'],lifecycle['accepted_objective'],rel_tol=0.0,abs_tol=0.0)
        for f,suffix in {'vp':'pi','vs':'mu','rho':'rho','qp':'qp','qs':'qs'}.items():
            readback=_read_floats(reload/'model'/f'current.fdveps.{suffix}',config.cell_count)
            assert readback.tobytes()==accepted[f].tobytes(),f
        report[backend]=dict(history=history,reload_objective=reload_result['objective'],
                             accepted_objective=lifecycle['accepted_objective'])
    (e['root']/f'{backend}_stages.json').write_text(json.dumps(report,indent=2))


@pytest.mark.parametrize('backend',[
    'cpu',pytest.param('cuda',marks=pytest.mark.optional_prerequisite)])
@pytest.mark.parametrize('filtered',[True,False])
@pytest.mark.parametrize('setting,parameter,value',[
    ('N_ORDER',False,1),('TIMEWIN',False,1),('NORMALIZE',False,1),
    ('OFFSET_MUTE',False,1),('TRKILL',True,1),('TIMELAPSE',True,1),
    ('QUELLTYPB',True,5),('SEISMO',True,2),('SEISMO',True,3),
])
def test_filtered_residual_modifiers_fail_closed(
        experiment,repository_root,denise_binary,mpiexec,request,backend,setting,parameter,value,filtered):
    e=experiment; case=e['root']/f'reject_{backend}_{setting}_{value}_{int(filtered)}'
    shots=(0,) if setting=='SEISMO' and filtered else (0,1)
    prepare(case,e['config'],e['model'],observed=e['truth'],shots=shots,filtered=filtered)
    field='TIME_WIN' if setting=='TIMEWIN' else setting
    (_set_parameters if parameter else _set_workflow)(case,**{field:value})
    binary=denise_binary if backend=='cpu' else request.getfixturevalue('cuda_denise_binary')
    with pytest.MonkeyPatch.context() as env:
        env.setenv('DENISE_PSV_BACKEND',backend)
        result=run_denise(repository_root=repository_root,case_directory=case,
            denise_binary=binary,mpiexec=mpiexec,ranks=1,
            configuration={'gate':'R7-filtered-residual-guard','setting':setting},
            timeout_seconds=120)
    assert result.returncode!=0,result_summary(result)
    output=result.stdout_path.read_text()+result.stderr_path.read_text()
    guard='transformed/weighted residuals are not qualified' in output
    earlier_cuda_envelope=(backend=='cuda' and setting=='SEISMO' and
        'violates frozen envelope' in output and f'SEISMO={value}' in output)
    assert guard or earlier_cuda_envelope,result_summary(result)
    assert not list((case/'jacobian').glob('gradient.raw.*'))

@pytest.mark.parametrize('backend',[
    'cpu',pytest.param('cuda',marks=pytest.mark.optional_prerequisite)])
def test_filtered_residual_guard_is_rechecked_on_second_stage(
        experiment,repository_root,denise_binary,mpiexec,request,backend):
    e=experiment; case=e['root']/f'reject_second_stage_{backend}'
    prepare(case,e['config'],e['model'],observed=e['truth'],stages=(12.,18.))
    path=case/'workflow.inp'; lines=path.read_text().splitlines()
    row=lines[2].split();row[lines[0].split().index('N_ORDER')]='1'
    lines[2]=' '.join(row);path.write_text('\n'.join(lines)+'\n')
    binary=denise_binary if backend=='cpu' else request.getfixturevalue('cuda_denise_binary')
    with pytest.MonkeyPatch.context() as env:
        env.setenv('DENISE_PSV_BACKEND',backend)
        result=run_denise(repository_root=repository_root,case_directory=case,
            denise_binary=binary,mpiexec=mpiexec,ranks=1,
            configuration={'gate':'R7-workflow-guard'},timeout_seconds=120)
    assert result.returncode!=0,result_summary(result)
    output=result.stdout_path.read_text()+result.stderr_path.read_text()
    assert 'transformed/weighted residuals are not qualified' in output,result_summary(result)
    lifecycle=json.loads((case/'jacobian'/'gradient.active_fwi.json').read_text())
    assert lifecycle['accepted_objective']<lifecycle['base_objective']
    assert 'stage_1_' in lifecycle['persisted_prefix']

@pytest.mark.parametrize('backend',[
    'cpu',pytest.param('cuda',marks=pytest.mark.optional_prerequisite)])
def test_multishot_ignores_zero_testshot_increment(
        experiment,repository_root,denise_binary,mpiexec,request,backend):
    e=experiment;case=e['root']/f'zero_testshot_increment_{backend}'
    prepare(case,e['config'],e['model'],observed=e['truth'])
    _set_parameters(case,TESTSHOTS='1,1,0')
    binary=denise_binary if backend=='cpu' else request.getfixturevalue('cuda_denise_binary')
    result=run(case,repository_root,binary,mpiexec,backend=backend,gradient_only=False)
    lifecycle=json.loads((case/'jacobian'/'gradient.active_fwi.json').read_text())
    assert lifecycle['accepted_objective']<lifecycle['base_objective']
    assert math.isclose(lifecycle['base_objective'],e['multi']['objective'],rel_tol=1e-14,abs_tol=0.)


@pytest.mark.parametrize('backend',[
    'cpu',pytest.param('cuda',marks=pytest.mark.optional_prerequisite)])
def test_filtered_single_shot_uses_complete_set_despite_testshot_selection(
        experiment,repository_root,denise_binary,mpiexec,request,backend):
    e=experiment;case=e['root']/f'filtered_single_excluded_selection_{backend}'
    prepare(case,e['config'],e['model'],observed=e['truth'],shots=(0,))
    _set_parameters(case,TESTSHOTS='2,2,1')
    binary=denise_binary if backend=='cpu' else request.getfixturevalue('cuda_denise_binary')
    run(case,repository_root,binary,mpiexec,backend=backend,gradient_only=False)
    lifecycle=json.loads((case/'jacobian'/'gradient.active_fwi.json').read_text())
    assert lifecycle['base_objective']==e['singles'][0]['objective']
    assert lifecycle['trial_forwards_min']>=1
    assert lifecycle['accepted_objective']<lifecycle['base_objective']
