from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.applications.test_denise_case import _inputs, _manifest, _su_trace
from tools.denise_case.config import load_case
from tools.denise_case.coordinates import coordinate_contract, pixel_coordinate
from tools.denise_case.generate import materialize_run
from tools.denise_case.rtm import (CORE_SHA, BENCHMARK_SHA, lame_parameters, stress_source,
    validated_component, mode2_records, materialize_rtm, read_raw_image, require_lane, material_envelope)
from tools.denise_case.rtm_qc import write_derived
from tools.denise_case.runner import _output_inventory
from tools.denise_case.validation import validate_case


def fixture_case(tmp_path):
    c = load_case(_manifest(tmp_path / 'fd4'))
    c.data['case']['id'] = 'A2-M9e4-FD4'
    c.data['physics'] = {'fd_order':4, 'max_relative_error':0}
    c.data['provenance']['denise_core_sha'] = CORE_SHA
    c.data['provenance']['benchmark_sha'] = BENCHMARK_SHA
    c.data['rtm'] = {'baseline_root':'../fd8', 'executable':'bin/denise_cuda'}
    c.path.write_bytes(c.canonical_bytes())
    _inputs(c.root)
    (c.root / 'input/geometry/sources.dat').write_text('1\n10 0 20 0 10 1 0 1\n')
    return c


def forward_fixture(c):
    run = materialize_run(c)
    for component in ('vx','vy'):
        samples = [float(i) for i in range(10)]
        blob = _su_trace(shot=1,receiver=2,samples=samples) + _su_trace(shot=1,receiver=1,samples=[-x for x in samples])
        (run / 'data' / f'observed_{component}.su.shot1').write_bytes(blob)
    signal = _su_trace(shot=1,receiver=1,samples=[float(i) for i in range(10)])
    (run / 'model/true_source_signal.0.su.shot1').write_bytes(signal)
    (run / 'run_metadata.json').write_text(json.dumps({'returncode':0,'output_inventory':_output_inventory(run)}))
    return run


def test_fd4_generation_and_fd8_lane_are_isolated(tmp_path):
    c = fixture_case(tmp_path)
    baseline = tmp_path / 'fd8'
    baseline.mkdir()
    (baseline / 'sentinel').write_bytes(b'untouched FD8')
    run = materialize_run(c)
    first = (run / 'denise.inp').read_bytes()
    assert materialize_run(c) == run
    assert first == (run / 'denise.inp').read_bytes()
    assert b'FD_ORDER =4' in first and b'MAX_RELATIVE_ERROR =0' in first
    assert (baseline / 'sentinel').read_bytes() == b'untouched FD8'
    with pytest.raises(ValueError,match='separate'):
        require_lane(load_case(_manifest(tmp_path / 'other')))


def test_mode2_records_are_deterministic_and_exact_envelope(tmp_path):
    c = fixture_case(tmp_path)
    assert mode2_records(c) == mode2_records(c)
    assert len(mode2_records(c)) == 122
    for line in ('MODE =2','QUELLART =3','INVMAT1 =3','NDT =1','DTINV =1','NPROCX =1','NPROCY =1','MAX_RELATIVE_ERROR =0'):
        assert line in mode2_records(c)
    c.data['physics']['max_relative_error'] = 1
    with pytest.raises(ValueError,match='Taylor'):
        mode2_records(c)


def test_lame_model_preparation_known_values_and_order():
    vp = np.array([[3,4],[5,6]],np.float32)
    vs = np.array([[0,2],[3,4]],np.float32)
    rho = np.array([[2,3],[4,5]],np.float32)
    lam,mu = lame_parameters(vp,vs,rho)
    np.testing.assert_array_equal(mu,[[0,12],[36,80]])
    np.testing.assert_array_equal(lam,[[18,24],[28,20]])
    assert lam.dtype == mu.dtype == np.float32
    with pytest.raises(ValueError,match='shapes'):
        lame_parameters(vp,vs[:,0],rho)


def test_source_preparation_including_endpoints_and_no_half_factor():
    # psource.c 1-based: s[2]/DT, (s[t+1]-s[t-1])/DT, -s[NT-1]/DT.
    np.testing.assert_array_equal(stress_source(np.array([1,3,8,9],np.float32),.5),[6,14,12,-16])
    assert stress_source(np.ones(5,np.float32),1).dtype == np.float32
    with pytest.raises(ValueError):
        stress_source(np.array([np.nan,1]),1)


def test_time_major_component_uses_coordinate_identity_not_trace_order(tmp_path):
    c = fixture_case(tmp_path)
    run = forward_fixture(c)
    from tools.denise_case.geometry import read_sources, read_receivers
    matrix = validated_component(run / 'data/observed_vx.su.shot1','vx',
        read_sources(c.root/'input/geometry/sources.dat')[0],read_receivers(c.root/'input/geometry/receivers.dat'),10,.001)
    assert matrix.shape == (10,2)
    assert matrix.dtype == np.dtype('<f4')
    np.testing.assert_array_equal(matrix[:3],[[0,0],[-1,1],[-2,2]])


def test_component_rejects_duplicates_and_inconsistent_source(tmp_path):
    c = fixture_case(tmp_path)
    from tools.denise_case.geometry import read_sources, read_receivers
    source = read_sources(c.root/'input/geometry/sources.dat')[0]
    receivers = read_receivers(c.root/'input/geometry/receivers.dat')
    path = tmp_path / 'bad.su.shot1'
    blob = _su_trace(shot=1,receiver=1,samples=[0]*10)
    path.write_bytes(blob+blob)
    with pytest.raises(ValueError,match='duplicate'):
        validated_component(path,'vx',source,receivers,10,.001)


def test_full_bridge_roundtrip_provenance_and_no_raw_overwrite(tmp_path):
    c = fixture_case(tmp_path)
    forward = forward_fixture(c)
    before = {p:p.read_bytes() for p in forward.rglob('*') if p.is_file()}
    run = materialize_rtm(c,[1],'preflight')
    assert before == {p:p.read_bytes() for p in forward.rglob('*') if p.is_file()}
    prepared = np.fromfile(run/'prepared/observed.vx.shot_1.bin',dtype='<f4').reshape(10,2)
    np.testing.assert_array_equal(prepared[1],[-1,1])
    source = np.fromfile(run/'prepared/source.shot_1.bin',dtype='<f4')
    np.testing.assert_array_equal(source,stress_source(np.arange(10,dtype=np.float32),.001))
    provenance = json.loads((run/'provenance.json').read_text())
    assert provenance['core_sha'] == CORE_SHA
    assert provenance['bridge'][0]['authoritative_shot'] == 1
    assert provenance['bridge'][0]['components']['vx']['prepared']['sha256']
    assert provenance['generator']['source_sha256']['tools/denise_case/model_io.py']
    raw = run/'raw/migration.image_lambda_raw.bin'
    raw.write_bytes(b'protected')
    with pytest.raises(FileExistsError,match='raw'):
        materialize_rtm(c,[1],'preflight')
    assert raw.read_bytes() == b'protected'


def test_raw_fp64_row_major_reader_and_invalid_size(tmp_path):
    values = np.arange(12,dtype='<f8').reshape(3,4)
    path = tmp_path / 'raw.bin'
    path.write_bytes(values.tobytes())
    np.testing.assert_array_equal(read_raw_image(path,4,3),values)
    assert read_raw_image(path,4,3).dtype == np.float64
    path.write_bytes(values.tobytes()[:-1])
    with pytest.raises(ValueError,match='size'):
        read_raw_image(path,4,3)


def test_raw_reader_rejects_nonfinite(tmp_path):
    path = tmp_path/'bad.bin'
    path.write_bytes(np.array([np.inf],dtype='<f8').tobytes())
    with pytest.raises(ValueError,match='nonfinite'):
        read_raw_image(path,1,1)


def test_derived_artifacts_are_separate_and_never_overwrite_raw(tmp_path):
    raw = tmp_path/'raw/migration.image_mu_raw.bin'
    raw.parent.mkdir()
    raw.write_bytes(b'raw sentinel')
    out = write_derived(tmp_path,'view.npy',np.ones((2,3)))
    assert out.parent == (tmp_path/'derived').resolve()
    assert raw.read_bytes() == b'raw sentinel'
    with pytest.raises(ValueError,match='safe'):
        write_derived(tmp_path,'../raw/migration.image_mu_raw.bin',np.ones(3))
    with pytest.raises(FileExistsError):
        write_derived(tmp_path,'view.npy',np.zeros((2,3)))


def test_coordinate_centers_edges_and_exact_overlay():
    contract = coordinate_contract(500,174,20)
    assert contract['first_sample_xy_m'] == [20,20]
    assert contract['last_sample_xy_m'] == [10000,3480]
    assert contract['imshow_extent_m'] == [10,10010,3490,10]
    assert pixel_coordinate(20,20,20) == (0,0)
    assert pixel_coordinate(10000,3480,20) == (499,173)
    assert pixel_coordinate(800,460,20) == (39,22)


def test_actual_fd4_taylor_dispersion_is_reported(tmp_path):
    c = fixture_case(tmp_path)
    report = validate_case(c,require_executable=False)
    assert report.facts['time_and_accuracy']['dispersion_fmax_hz'] == pytest.approx(500/80)
    assert any('FD4/error-class-0' in w for w in report.warnings)
    assert not any('FD8' in w for w in report.warnings)


def test_required_model_io_source_is_visible_to_git():
    root = Path(__file__).resolve().parents[2]
    check = subprocess.run(['git','check-ignore','tools/denise_case/model_io.py'],cwd=root,capture_output=True,text=True)
    assert check.returncode == 1, check.stdout
    assert subprocess.run(['git','check-ignore','model_test.bin'],cwd=root,capture_output=True).returncode == 0


def test_material_envelope_reports_water_without_modifying_it():
    vp = np.full((2,3),1500,np.float32)
    vs = np.zeros((2,3),np.float32)
    rho = np.full((2,3),1010,np.float32)
    lam,mu = lame_parameters(vp,vs,rho)
    before = (lam.tobytes(),mu.tobytes(),rho.tobytes())
    report = material_envelope(lam,mu,rho)
    assert not report['supported']
    assert report['zero_mu_cells'] == report['invalid_cells'] == 6
    assert report['first_invalid_y_x'] == [0,0]
    assert before == (lam.tobytes(),mu.tobytes(),rho.tobytes())


def test_cuda_spec_default_delegates_to_make_without_execution(monkeypatch):
    from tools.denise_case.rtm import cuda_build_spec
    def forbidden(*args,**kwargs):
        raise AssertionError('build specification must not execute make/compiler/probes')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    result=cuda_build_spec(environ={})
    assert result['command']==['make','-C','src','denise_cuda','-j4']
    assert result['toolkit_probe']==['nvcc','--version']
    assert result['compiler_override'] is None and result['architecture_override'] is None
    assert result['compiler_selection']=='make/PATH default'
    assert result['architecture_selection']=='repository make default'
    assert result==cuda_build_spec(environ={})


def test_cuda_spec_explicit_compiler_path_and_environment_precedence():
    from tools.denise_case.rtm import cuda_build_spec
    compiler='caller-selected/toolkit/bin/nvcc'
    result=cuda_build_spec(nvcc=compiler,cuda_archs='90',environ={'NVCC':'other-nvcc','CUDA_ARCHS':'86'})
    assert result['command']==['make','-C','src','denise_cuda',f'NVCC={compiler}','CUDA_ARCHS=90','-j4']
    assert result['toolkit_probe']==[compiler,'--version']
    assert result['compiler_selection']==result['architecture_selection']=='explicit'
    assert result==cuda_build_spec(nvcc=compiler,cuda_archs='90',environ={'NVCC':'ignored','CUDA_ARCHS':'75'})


@pytest.mark.parametrize('archs',['86','90','86 90'])
def test_cuda_spec_caller_architectures_propagate_without_toolkit_assumption(archs):
    from tools.denise_case.rtm import cuda_build_spec
    result=cuda_build_spec(cuda_archs=archs,environ={})
    assert f'CUDA_ARCHS={archs}' in result['command']
    assert not any(v.startswith('NVCC=') for v in result['command'])
    assert result['toolkit_probe']==['nvcc','--version']


def test_cuda_spec_environment_overrides_are_intentionally_supported():
    from tools.denise_case.rtm import cuda_build_spec
    result=cuda_build_spec(environ={'NVCC':'alternative-nvcc','CUDA_ARCHS':'75 90'})
    assert result['command']==['make','-C','src','denise_cuda','NVCC=alternative-nvcc','CUDA_ARCHS=75 90','-j4']
    assert result['compiler_selection']==result['architecture_selection']=='environment'


@pytest.mark.parametrize('archs',['','86;90','$(machine-specific-detection)'])
def test_cuda_spec_invalid_architecture_fails_without_build(archs):
    from tools.denise_case.rtm import cuda_build_spec
    with pytest.raises(ValueError,match='architectures'):
        cuda_build_spec(cuda_archs=archs,environ={})


def test_cuda_spec_invalid_compiler_fails_without_build():
    from tools.denise_case.rtm import cuda_build_spec
    with pytest.raises(ValueError,match='NVCC'):
        cuda_build_spec(nvcc='',environ={})


def test_cuda_cli_selection_reaches_build_api_spec_only(monkeypatch,capsys):
    from tools.denise_case import cli
    from tools.denise_case.rtm import cuda_build_spec
    sentinel=object(); received=[]
    monkeypatch.setattr(cli,'load_case',lambda path:sentinel)
    # Replace the build API with ONLY the pure specification for routing proof;
    # this does not simulate or claim a successful compilation.
    def specification_only(config,**kwargs):
        assert config is sentinel
        received.append(kwargs)
        return cuda_build_spec(**kwargs,environ={})
    monkeypatch.setattr(cli,'build_cuda',specification_only)
    def forbidden(*args,**kwargs):
        raise AssertionError('no build or compiler execution is allowed')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    assert cli.main(['build-cuda','--nvcc','caller-nvcc','--cuda-archs','90'])==0
    assert received==[{'nvcc':'caller-nvcc','cuda_archs':'90'}]
    assert json.loads(capsys.readouterr().out)['command']==['make','-C','src','denise_cuda','NVCC=caller-nvcc','CUDA_ARCHS=90','-j4']
    assert cli.main(['build-cuda'])==0
    assert received[-1]=={'nvcc':None,'cuda_archs':None}
    assert json.loads(capsys.readouterr().out)['command']==['make','-C','src','denise_cuda','-j4']
