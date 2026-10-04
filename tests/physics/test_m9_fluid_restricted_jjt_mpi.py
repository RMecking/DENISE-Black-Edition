"""CPU MPI actual owned halos, restricted transpose and collective rejection."""
from dataclasses import replace
import json
import os
import subprocess
import numpy as np
import pytest
from tests.physics import test_m9d2_elastic_psv_mpi as mpi
from tests.physics import test_m9b1_elastic_psv_born_production as accepted
from tests.utilities import zero_shear_reference as z
from tests.utilities import m9_fluid_restricted_reference as ref


@pytest.fixture(scope="session")
def restricted_mpi(tmp_path_factory,repository_root):
    directory=tmp_path_factory.mktemp("fluid2-mpi")
    exe=directory/"operator"
    subprocess.run(["mpicc","-std=c99","-O2","-Wall","-Wextra","-Werror","-pedantic",
        "-I"+str(repository_root/"include"),
        str(repository_root/"tests/utilities/m9_fluid_restricted_mpi.c"),
        str(repository_root/"src/PSV/elastic_psv_born_mpi.c"),
        "-Wl,--wrap=calloc","-lm","-o",str(exe)],check=True)
    return exe,directory


@pytest.mark.parametrize("top",mpi.TOPS)
@pytest.mark.parametrize("pattern,fs,fw",[
    ("horizontal",0,0),("offset",0,0),("vertical",0,3),
    ("horizontal",1,0),("offset",1,3)])
def test_actual_distributed_restricted_graph(restricted_mpi,top,pattern,fs,fw):
    exp=ref.fixture(pattern,fs,fw);m=z.material(exp.rho,exp.lam,exp.mu)
    dl,dm=(a.astype(np.float32).astype(float) for a in z.directions(m)["joint"])
    data=np.random.default_rng(973).normal(size=(exp.nt,len(exp.receivers),2)).astype(np.float32)
    expected=ref.products(exp,ref.b._source_samples(exp).astype(np.float32).astype(float),dl,dm,data.astype(float),fs)
    outputs=[]
    for segments in (0,3):
        # Input path is created by mpi.run; each wrapper reads only the same
        # test fixture file to pick exactly one owned invalid dMu entry.
        exe,root=restricted_mpi
        label=f"{pattern}-{fs}-{fw}-{top}-{segments}"
        predicted=root/f"{label}-{len(list(root.iterdir()))}"
        directory,p=mpi.run(restricted_mpi,exp,dl,dm,data,top,segments,
            env={"M9D3_FREE_SURF":str(fs),"FLUID2_TRANSACTIONS":"1",
                 "FLUID2_INPUT":str(predicted/"input.bin")},label=label)
        assert "FLUID2_MPI_INVALID_SINGLE_OWNER_BEFORE_MUTATION_PASS" in p.stdout
        assert "FLUID2_MPI_BOTH_TOPOLOGY_TRIALS_TRANSACTION_RECOVERY_PASS" in p.stdout
        actual=mpi.read(directory,exp)
        raw=(actual["gl"],actual["gm"])
        np.testing.assert_array_equal(raw[1][m.fluid],0.)
        assert not np.signbit(raw[1][m.fluid]).any()
        if fs:
            np.testing.assert_array_equal(raw[0][0,m.fluid[0]],0.)
            assert not np.signbit(raw[0][0,m.fluid[0]]).any()
        metric=accepted._production_dot_metrics(actual["j"],data,dl,dm,*raw,*expected)
        accepted._assert_production_dot_closes(metric)
        analytic=actual["j"].astype(float)
        finite=(np.fromfile(directory/"plus.bin",np.float32).astype(float)-
                np.fromfile(directory/"minus.bin",np.float32).astype(float)).reshape(analytic.shape)/.1
        fd=float(np.linalg.norm(finite-analytic)/np.linalg.norm(analytic))
        assert fd<=.004,fd
        outputs.append(directory)
        ref.record("mpi",top=list(top),pattern=pattern,fs=fs,fw=fw,segments=segments,
            fd_relative=fd,transactional=True,serial_metrics={k:v for k,v in actual.items() if k.endswith("_metrics")},**metric)
    for name in ("background","j","gl","gm"):
        assert (outputs[0]/(name+".bin")).read_bytes()==(outputs[1]/(name+".bin")).read_bytes()


@pytest.fixture(scope="session")
def solid_base_mpi(tmp_path_factory,repository_root):
    import hashlib
    from pathlib import Path
    source=os.environ.get("DENISE_FLUID2_BASE_MPI_SOURCE")
    if not source:pytest.skip("exact BASE MPI source artifact required for bitwise provenance gate")
    source=Path(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest()=="e9fdc9f1f357eb170c44996bcc9117ff4b2fab7e1f16012cb3da6667f7e396c7"
    directory=tmp_path_factory.mktemp("fluid2-solid-base-mpi");exe=directory/"operator"
    subprocess.run(["mpicc","-std=c99","-O2","-Wall","-Wextra","-Werror","-pedantic",
        "-I"+str(repository_root/"include"),
        str(repository_root/"tests/utilities/m9d2_elastic_psv_mpi_harness.c"),
        str(source),"-Wl,--wrap=calloc","-lm","-o",str(exe)],check=True)
    return exe,directory


from tests.physics.test_m9d2_elastic_psv_mpi import mpi_harness


@pytest.mark.parametrize("fs,fw",[(0,0),(0,3),(1,0),(1,3)])
@pytest.mark.parametrize("segments",[0,3])
def test_all_solid_mpi_base_bitwise(mpi_harness,solid_base_mpi,fs,fw,segments):
    exp=ref.fixture("solid",fs,fw);m=z.material(exp.rho,exp.lam,exp.mu)
    dl,dm=z.directions(m)["joint"]
    data=np.random.default_rng(973).normal(size=(exp.nt,len(exp.receivers),2)).astype(np.float32)
    outputs=[]
    for harness in (mpi_harness,solid_base_mpi):
        directory,_=mpi.run(harness,exp,dl,dm,data,(2,2),segments,env={"M9D3_FREE_SURF":str(fs)},label="solid-base")
        mpi.read(directory,exp);outputs.append(directory)
    for name in ("background","j","gl","gm","plus","minus","strain0","strain1","strain2","strain3"):
        assert (outputs[0]/(name+".bin")).read_bytes()==(outputs[1]/(name+".bin")).read_bytes()
    ref.record("solid_mpi_base",fs=fs,fw=fw,segments=segments,top=[2,2],bitwise=True,
               BASE="d66437ebe37bff82d95e24c1bff6d79f4e29660e")
