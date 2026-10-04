"""Actual mixed-corner products, copied classification and exact BASE solids."""
import ctypes as C
import hashlib
import itertools
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from tests.physics import test_m9_fluid_cpu_forward as f
from tests.physics.test_m9_fluid_cpu_forward import fluid_library
from tests.physics import test_m9_fluid_restricted_jjt as wave
from tests.utilities import zero_shear_reference as z
from tests.utilities import m9_fluid_restricted_reference as ref


@pytest.fixture(scope="session")
def material_library(tmp_path_factory,repository_root):
    output=tmp_path_factory.mktemp("fluid2-material")/"products.so"
    subprocess.run(["cc","-std=c99","-O2","-Wall","-Wextra","-Werror","-pedantic","-fPIC","-shared",
        "-I"+str(repository_root/"include"),
        str(repository_root/"tests/utilities/m9_fluid_restricted_material.c"),
        "-lm","-o",str(output)],check=True)
    api=f.library(output);api.fluid2_material_products.argtypes=[f.P,f.F,f.D,f.F,f.D]
    return api


@pytest.mark.parametrize("occupancy",list(itertools.product((0,1),repeat=4)))
def test_corner_jvp_vjp_all_occupancy_before_division(material_library,occupancy):
    mu=np.full((8,10),3.)
    mu[3:5,4:6]=np.array([2,3,5,7]).reshape(2,2)*(1-np.array(occupancy).reshape(2,2))
    c=f.Context(material_library,mu=mu,nt=9)
    try:
        m=z.material(c.rho,c.lam,c.mu)
        dm=np.cos(np.indices(c.shape).sum(axis=0)).astype(np.float32);dm[m.fluid]=0.
        bar=np.zeros(c.shape);bar[3,4]=1.
        dh=np.empty(c.shape,np.float32);gm=np.empty(c.shape)
        c.api.fluid_clear_arithmetic()
        c.check(c.api.fluid2_material_products(c.c,f.fp(dm),f.dp(bar),f.fp(dh),f.dp(gm)))
        expected=z.corner_jvp(m,dm);expected_g=z.corner_vjp(m,bar)
        assert np.linalg.norm(dh-expected)/np.linalg.norm(expected)<=1e-5
        if any(occupancy):
            assert dh[3,4]==0 and not np.signbit(dh[3,4])
            np.testing.assert_array_equal(gm,0.)
            assert not np.signbit(gm).any()
        else:
            assert np.linalg.norm(gm-expected_g)/np.linalg.norm(expected_g)<=6e-5
        assert c.api.fluid_singular_arithmetic()==0
    finally:c.close()


def test_copied_classification_and_positive_subnormal(fluid_library):
    mu=np.ones((20,24));mu[0,0]=0.;mu[1,1]=np.float32(1e-40)
    c=f.Context(fluid_library,mu=mu,nt=9)
    try:
        maps=c.maps();c.mu[0,0]=1.;c.prepare()
        np.testing.assert_array_equal(c.maps(),maps)
        assert maps[2,1,1]>0
        dl=np.zeros(c.shape,np.float32);dm=dl.copy();dm[0,0]=1.
        out=np.full(c.data_shape,19,np.float32);before=c.snapshot()
        assert c.api.denise_elastic_psv_born_apply_j(c.c,f.fp(dl),f.fp(dm),f.fp(out))!=0
        assert np.all(out==19) and c.snapshot()==before
        dm.fill(0);np.testing.assert_array_equal(wave.j(c,dl,dm),0.)
        # Positive copied subnormal is a solid parameter, even after caller
        # changes that buffer to zero. No tolerance or mask is consulted.
        c.mu[1,1]=0.;dm[1,1]=1.
        result=wave.j(c,dl,dm)
        assert np.isfinite(result).all()
    finally:c.close()


@pytest.fixture(scope="session")
def base_library(tmp_path_factory,repository_root):
    source=os.environ.get("DENISE_FLUID2_BASE_SOURCE")
    if not source:pytest.skip("exact BASE source artifact required for bitwise provenance gate")
    source=Path(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest()=="f76dce894c3e761bf0f911c827d40f1304fedfba7c3cf8f8a91f5ac36a163eb1"
    out=tmp_path_factory.mktemp("fluid2-base")/"base.so"
    subprocess.run(["cc","-std=c99","-O2","-Wall","-Wextra","-Werror","-pedantic","-fPIC","-shared",
        "-I"+str(repository_root/"include"),'-DDENISE_FLUID_SERIAL_SOURCE="'+str(source)+'"',
        str(repository_root/"tests/utilities/m9_fluid_cpu_products.c"),"-lm","-o",str(out)],check=True)
    return f.library(out)


@pytest.mark.parametrize("fs,fw",[(0,0),(0,3),(1,0),(1,3)])
@pytest.mark.parametrize("segments",[0,3])
def test_all_solid_base_bitwise(fluid_library,base_library,fs,fw,segments):
    exp=ref.fixture("solid",fs,fw)
    a=wave.context(fluid_library,exp,fs,segments);b=wave.context(base_library,exp,fs,segments)
    try:
        dl,dm=z.directions(z.material(exp.rho,exp.lam,exp.mu))["joint"]
        data=np.random.default_rng(973).normal(size=a.data_shape).astype(np.float32)
        for x,y in ((a.prepare(),b.prepare()),(a.maps(),b.maps()),
            (wave.j(a,dl,dm),wave.j(b,dl,dm)),(wave.jt(a,data),wave.jt(b,data))):
            assert x.tobytes()==y.tobytes()
        for x,y in zip(a.products()[:3],b.products()[:3]):assert x.tobytes()==y.tobytes()
        ref.record("solid_base",fs=fs,fw=fw,segments=segments,bitwise=True,
                   BASE="d66437ebe37bff82d95e24c1bff6d79f4e29660e")
    finally:a.close();b.close()
