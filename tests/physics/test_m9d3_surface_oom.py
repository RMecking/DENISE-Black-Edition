"""Discover and inject every FREE_SURF reachable allocation, including scratch."""
import pytest
from tests.physics.test_m9d2_h1_collective_oom_cleanup import oom_ledger,run,PHASES,MATRIX

@pytest.mark.parametrize('phase',PHASES)
@pytest.mark.parametrize('px,py,segments',MATRIX)
def test_surface_all_reachable_allocations_release(oom_ledger,tmp_path,monkeypatch,phase,px,py,segments):
    monkeypatch.setenv('M9D3_FREE_SURF','1')
    evidence=run(oom_ledger,tmp_path,phase,(px,py),segments,0,-1)
    counts=[r['calls'] for r in evidence['cases'][0]['ranks']]
    expected={(r,i) for r,n in enumerate(counts) for i in range(1,n+1)}
    got={(c['target'],c['index']) for c in evidence['cases'][1:]}
    assert expected==got
    assert len(evidence['cases'])==1+sum(counts)
    print('FP32/FP64_PRODUCTION_SURFACE_OOM',phase,(px,py),segments,counts,len(expected))
