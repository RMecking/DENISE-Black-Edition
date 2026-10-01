"""Actual C=Bghost H Ztraction and C^T=Z^T H^T B^T unit-basis gates."""
import json
import subprocess
import numpy as np
import pytest
from tests.physics.test_m9d2_elastic_psv_mpi import TOPS
from tests.utilities.elastic_psv_free_surface_reference import halo_surface_maps

@pytest.fixture(scope='module')
def composition_binary(tmp_path_factory,repository_root):
    out=tmp_path_factory.mktemp('m9d3-composition')/'operator'
    subprocess.run(['mpicc','-std=c99','-O2','-Wall','-Wextra','-Werror','-pedantic',
                    '-I',str(repository_root/'include'),str(repository_root/'tests/utilities/m9d3_surface_mpi_blocks.c'),
                    '-lm','-o',str(out)],check=True)
    return out

@pytest.mark.parametrize('top',TOPS)
@pytest.mark.parametrize('kind',('syy','sxy'))
def test_actual_composed_halo_surface_unit_basis(composition_binary,tmp_path,top,kind):
    h,b=halo_surface_maps(8,8,*top,kind);size=h.size
    lx,ly=8//top[0],8//top[1];cells=(lx+4)*(ly+4)
    z=np.eye(size)
    if kind=='syy':
        for rank in range(top[0]):
            for i in range(lx):z[rank*cells+2*(lx+4)+i+2]=0
    expected=b.matrix()@h.matrix()@z
    with (tmp_path/'input.bin').open('wb') as f:
        np.array([size],np.int32).tofile(f)
        for i in range(size):
            x=np.zeros(size,np.float32);x[i]=1;x.tofile(f)
            y=np.zeros(size,np.float64);y[i]=1;y.tofile(f)
    p=subprocess.run(['mpiexec','--oversubscribe','-n',str(top[0]*top[1]),str(composition_binary),
                      str(tmp_path/'input.bin'),str(tmp_path/'output.bin'),str(top[0]),str(top[1]),str(3 if kind=='syy' else 4)],capture_output=True,text=True,timeout=120)
    (tmp_path/'run.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stdout+p.stderr
    columns=[];rows=[]
    with (tmp_path/'output.bin').open('rb') as f:
        for i in range(size):
            columns.append(np.fromfile(f,np.float32,size))
            rows.append(np.fromfile(f,np.float64,size))
        assert not f.read()
    actual=np.stack(columns,axis=1);reverse=np.stack(rows,axis=1)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(reverse,expected.T)
    rng=np.random.default_rng(418);x=rng.normal(size=size);y=rng.normal(size=size)
    residual=abs(np.dot(actual@x,y)-np.dot(x,reverse@y))/(np.linalg.norm(x)*np.linalg.norm(y))
    assert residual<5e-13
    metrics=dict(topology=top,kind=kind,basis_columns=size,forward_max_error=0,reverse_max_error=0,dot=residual)
    (tmp_path/'composition.json').write_text(json.dumps(metrics,indent=2))
    print('FP32/FP64_PRODUCTION_MPI_COMPOSITION',metrics)
