"""Requested-byte ownership gates; no change to any scientific M9d2 oracle."""
import json
import os
import shutil
import subprocess

import pytest

PHASES=('prepare','j','jt','checkpoint')
MATRIX=((2,1,0),(2,1,3),(1,2,3),(2,2,3))

@pytest.fixture(scope='session')
def oom_ledger(tmp_path_factory,repository_root):
    directory=tmp_path_factory.mktemp('m9d2-oom-ledger')
    executable=directory/'ledger'
    compiler=shutil.which('mpicc')
    assert compiler,'local MPI compiler required'
    flags=['-std=c99','-g','-O2','-Wall','-Wextra','-Werror','-pedantic']
    if os.environ.get('DENISE_M9D2_OOM_SANITIZE')=='1':
        flags+=['-O1','-fsanitize=address,undefined','-fno-omit-frame-pointer','-fno-pie','-no-pie']
    subprocess.run([compiler,*flags,'-I',str(repository_root/'include'),
                    str(repository_root/'tests/utilities/m9d2_oom_ledger.c'),
                    str(repository_root/'src/PSV/elastic_psv_born_mpi.c'),
                    '-Wl,--wrap=calloc','-Wl,--wrap=free','-lm','-o',str(executable)],check=True)
    return executable

def run(oom_ledger,tmp_path,phase,top,segments,target,index):
    result=tmp_path/'ledger.json'
    env=os.environ.copy()
    if env.get('DENISE_M9D2_OOM_SANITIZE')=='1':
        env.update(ASAN_OPTIONS='detect_leaks=0:abort_on_error=1',
                   UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    completed=subprocess.run(['mpiexec','--oversubscribe','-n',str(top[0]*top[1]),
                             str(oom_ledger),phase,str(top[0]),str(top[1]),
                             str(segments),str(target),str(index),str(result)],
                            env=env,capture_output=True,text=True,timeout=120)
    (tmp_path/'run.log').write_text(completed.stdout+completed.stderr)
    assert completed.returncode==0,completed.stdout+completed.stderr
    evidence=json.loads(result.read_text())
    assert evidence['pass']
    for case in evidence['cases']:
        assert case['pass']
        for rank in case['ranks']:
            assert rank['outstanding_bytes']==rank['outstanding_allocations']==0
            assert rank['invalid_free']==0 and rank['destroyed'] and rank['output_ok']
            assert rank['status']!=0 if case['index'] else rank['status']==0
            assert rank['injected']==int(bool(case['index']) and rank['rank']==case['target'])
            assert rank['prepared']==(int(case['index']==0) if phase=='prepare' else 1)
    return evidence

@pytest.mark.parametrize('phase',PHASES)
@pytest.mark.parametrize('px,py,segments',MATRIX)
def test_all_reachable_allocations_collectively_release(oom_ledger,tmp_path,phase,px,py,segments):
    evidence=run(oom_ledger,tmp_path,phase,(px,py),segments,0,-1)
    counts=[r['calls'] for r in evidence['cases'][0]['ranks']]
    expected={(rank,index) for rank,count in enumerate(counts) for index in range(1,count+1)}
    actual={(case['target'],case['index']) for case in evidence['cases'][1:]}
    assert actual==expected
    assert len(evidence['cases'])==1+sum(counts)

@pytest.mark.parametrize('phase,index',[('prepare',20),('j',20),('jt',17),('checkpoint',1),('prepare',2)])
def test_original_peer_temporary_leaks_are_gone(oom_ledger,tmp_path,phase,index):
    # These indices belong only to this frozen 24x20/2x1/NT10 FULL fixture.
    # The exhaustive gate above discovers allocation counts from successful runs.
    run(oom_ledger,tmp_path,phase,(2,1),0,1,index)
