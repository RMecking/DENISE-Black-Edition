"""Small post-processing fixtures; no DENISE execution or core oracle changes."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tools.denise_case import campaign, fluid2
from tools.denise_case.model_io import sha256_file, write_grid


SHAPE=(3,2)


def fixture(root, ids=(1,2,10)):
    root.mkdir()
    vs=np.array([[0,0],[0,0],[1,1]],dtype=np.float32)
    receipt={'completed':len(ids),'failed':0,'per_shot':{}}
    for shot in reversed(ids):
        run=root/f'shot_{shot:03d}'; (run/'raw').mkdir(parents=True)
        (run/'provenance.json').write_bytes(b'{"fixture":true}\n')
        values=np.full(SHAPE,float(shot),dtype='<f8')
        mu=values.copy(); mu[:2]=0
        raw={}
        for key,array in (('lambda',values),('mu',mu)):
            path=run/f'raw/migration.image_{key}_raw.bin'; path.write_bytes(array.tobytes())
            raw[key]={'path':'/retired/host/layout/'+path.name,'sha256':sha256_file(path),
                      'bytes':array.nbytes,'shape_y_x':list(SHAPE),'dtype':'<f8','order':'row-major [y,x]'}
        metadata={'complete':True,'returncode':0,'backend':'CPU-M9','core_sha':fluid2.CORE_SHA,
                  'binary_sha256':'a'*64,'authoritative_shots':[shot],
                  'provenance_sha256':sha256_file(run/'provenance.json'),
                  'raw_images':raw,'qc':fluid2.image_checks(values,mu,vs),'runtime_seconds':float(shot)}
        path=run/'run_metadata.json'; path.write_text(json.dumps(metadata,sort_keys=True)+'\n')
        receipt['per_shot'][str(shot)]={'metadata':{'sha256':sha256_file(path)},
                                      'raw_lambda':raw['lambda'],'raw_mu':raw['mu']}
    manifest=root.parent/'campaign.json'; manifest.write_text(json.dumps(receipt,sort_keys=True)+'\n')
    model=root.parent/'smooth2.vs'; write_grid(model,vs)
    return manifest,model


def edit_metadata(root,shot,edit):
    path=root/f'shot_{shot:03d}/run_metadata.json'
    value=json.loads(path.read_text()); edit(value); path.write_text(json.dumps(value))


def protected(root):
    return {p.relative_to(root).as_posix():sha256_file(p) for p in root.rglob('*') if p.is_file()}


def test_aggregate_numeric_order_hash_receipt_and_input_protection(tmp_path):
    root=tmp_path/'runs'; manifest,vs=fixture(root)
    before=protected(root)
    result=campaign.aggregate_campaign(root,tmp_path/'aggregate',[10,1,2],shape=SHAPE,
                                        campaign_receipt=manifest,vs_model=vs)
    assert result['order']==[1,2,10]
    assert result['configuration']['accumulation_dtype']=='<f8'
    values=np.fromfile(tmp_path/'aggregate/raw/migration.image_lambda_raw.bin',dtype='<f8').reshape(SHAPE)
    np.testing.assert_array_equal(values,np.full(SHAPE,13.))
    for key in ('lambda','mu'):
        assert result['outputs'][key]['sha256']==sha256_file(tmp_path/'aggregate'/result['outputs'][key]['path'])
        assert result['input_hashes']['10'][key]==sha256_file(root/f'shot_010/raw/migration.image_{key}_raw.bin')
    assert result['qc']['water_mu_exact_positive_zero']
    assert protected(root)==before


def test_accumulation_really_uses_fp64_without_average_or_weights(tmp_path):
    root=tmp_path/'runs'; fixture(root)
    for shot,number in ((1,16777216.),(2,1.),(10,-16777216.)):
        path=root/f'shot_{shot:03d}/raw/migration.image_lambda_raw.bin'
        path.write_bytes(np.full(SHAPE,number,dtype='<f8').tobytes())
        edit_metadata(root,shot,lambda m:m['raw_images']['lambda'].update(sha256=sha256_file(path)))
    result=campaign.aggregate_campaign(root,tmp_path/'sum',[10,2,1],shape=SHAPE)
    np.testing.assert_array_equal(np.fromfile(tmp_path/'sum/raw/migration.image_lambda_raw.bin',dtype='<f8'),np.ones(6))
    assert not any(result['configuration'][k] for k in ('averaging','normalization','smoothing','mute',
                    'illumination_weighting','preconditioning','parameter_transform'))


@pytest.mark.parametrize('failure',['duplicate','missing','incomplete','wrong_core','wrong_binary','bad_provenance'])
def test_invalid_campaign_fails_closed_without_outputs(tmp_path,failure):
    root=tmp_path/'runs'; fixture(root)
    if failure=='duplicate':
        shutil.copytree(root/'shot_001',root/'shot_001_copy')
    elif failure=='missing':
        (root/'shot_002/run_metadata.json').unlink()
    elif failure=='incomplete':
        edit_metadata(root,1,lambda m:m.update(complete=False))
    elif failure=='wrong_core':
        edit_metadata(root,1,lambda m:m.update(core_sha='b'*40))
    elif failure=='wrong_binary':
        edit_metadata(root,1,lambda m:m.update(binary_sha256='b'*64))
    else:
        (root/'shot_001/provenance.json').write_bytes(b'changed')
    with pytest.raises(ValueError):
        campaign.aggregate_campaign(root,tmp_path/'never_written',[1,2,10],shape=SHAPE)
    assert not (tmp_path/'never_written').exists()


@pytest.mark.parametrize('field,value',[('dtype','<f4'),('shape_y_x',[2,3]),('order','x-major, y-fastest')])
def test_wrong_datatype_shape_or_layout_rejected(tmp_path,field,value):
    root=tmp_path/'runs'; fixture(root)
    edit_metadata(root,1,lambda m:m['raw_images']['lambda'].update({field:value}))
    with pytest.raises(ValueError,match='datatype/layout/shape'):
        campaign.aggregate_campaign(root,tmp_path/'bad',[1,2,10],shape=SHAPE)


def test_wrong_size_and_changed_raw_hash_rejected(tmp_path):
    root=tmp_path/'runs'; fixture(root)
    path=root/'shot_001/raw/migration.image_lambda_raw.bin'; original=path.read_bytes()
    path.write_bytes(original[:-1])
    with pytest.raises(ValueError,match='size'):
        campaign.aggregate_campaign(root,tmp_path/'bad',[1,2,10],shape=SHAPE)
    path.write_bytes(np.full(SHAPE,4.,dtype='<f8').tobytes())
    with pytest.raises(ValueError,match='hash changed'):
        campaign.aggregate_campaign(root,tmp_path/'bad',[1,2,10],shape=SHAPE)


def test_nonfinite_images_rejected_even_with_matching_hash(tmp_path):
    root=tmp_path/'runs'; fixture(root)
    path=root/'shot_001/raw/migration.image_lambda_raw.bin'
    path.write_bytes(np.full(SHAPE,np.inf,dtype='<f8').tobytes())
    edit_metadata(root,1,lambda m:m['raw_images']['lambda'].update(sha256=sha256_file(path)))
    with pytest.raises(ValueError,match='nonfinite'):
        campaign.aggregate_campaign(root,tmp_path/'bad',[1,2,10],shape=SHAPE)


def test_no_overwrite_and_no_output_inside_individual_raw_or_derived(tmp_path):
    root=tmp_path/'runs'; fixture(root)
    output=tmp_path/'aggregate'
    campaign.aggregate_campaign(root,output,[1,2,10],shape=SHAPE)
    before=protected(output)
    with pytest.raises(FileExistsError,match='overwrite'):
        campaign.aggregate_campaign(root,output,[1,2,10],shape=SHAPE)
    assert protected(output)==before
    with pytest.raises(ValueError,match='separate'):
        campaign.aggregate_campaign(root,root/'shot_001/derived/new',[1,2,10],shape=SHAPE)


def test_duplicate_expected_ids_and_campaign_receipt_hash_mismatch(tmp_path):
    root=tmp_path/'runs'; manifest,_=fixture(root)
    with pytest.raises(ValueError,match='duplicate expected'):
        campaign.aggregate_campaign(root,tmp_path/'bad',[1,1,2],shape=SHAPE)
    value=json.loads(manifest.read_text()); value['per_shot']['1']['metadata']['sha256']='0'*64
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='metadata hash differs'):
        campaign.aggregate_campaign(root,tmp_path/'bad',[1,2,10],shape=SHAPE,campaign_receipt=manifest)


def test_report_portable_deterministic_caveats_and_never_calls_solver(tmp_path,monkeypatch):
    import subprocess
    def forbidden(*args,**kwargs):
        raise AssertionError('reporting must not execute any subprocess')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    root=tmp_path/'runs'; manifest,vs=fixture(root)
    campaign.aggregate_campaign(root,tmp_path/'aggregate',[1,2,10],shape=SHAPE,campaign_receipt=manifest,vs_model=vs)
    args={'shape':SHAPE,'campaign_receipt':manifest,'vs_model':vs}
    before=protected(root)
    result=campaign.report_campaign(root,tmp_path/'report1',tmp_path/'aggregate/receipt.json',[1,2,10],**args)
    campaign.report_campaign(root,tmp_path/'report2',tmp_path/'aggregate/receipt.json',[1,2,10],**args)
    assert (tmp_path/'report1/summary.json').read_bytes()==(tmp_path/'report2/summary.json').read_bytes()
    assert (tmp_path/'report1/index.html').read_bytes()==(tmp_path/'report2/index.html').read_bytes()
    shutil.copytree(root,tmp_path/'relocated')
    campaign.report_campaign(tmp_path/'relocated',tmp_path/'report3',tmp_path/'aggregate/receipt.json',[1,2,10],**args)
    assert (tmp_path/'report1/summary.json').read_bytes()==(tmp_path/'report3/summary.json').read_bytes()
    text=(tmp_path/'report1/index.html').read_text()
    assert str(tmp_path) not in text and '/retired/host' not in text
    assert all(c in result['scientific_caveats'] for c in campaign.CAVEATS)
    assert result['water_mu_exact_positive_zero'] and result['water_mu_max_abs']==0
    assert result['solver_executions']==0 and result['observations_are_not_acceptance_thresholds']
    assert protected(root)==before
    with pytest.raises(FileExistsError):
        campaign.report_campaign(root,tmp_path/'report1',tmp_path/'aggregate/receipt.json',[1,2,10],**args)


@pytest.mark.parametrize('problem',['input_hash','order','output_hash','configuration'])
def test_report_rejects_aggregate_receipt_inconsistency(tmp_path,problem):
    root=tmp_path/'runs'; _,vs=fixture(root)
    campaign.aggregate_campaign(root,tmp_path/'aggregate',[1,2,10],shape=SHAPE,vs_model=vs)
    path=tmp_path/'aggregate/receipt.json'; value=json.loads(path.read_text())
    if problem=='input_hash': value['input_hashes']['1']['lambda']='0'*64
    elif problem=='order': value['order'].reverse()
    elif problem=='output_hash': value['outputs']['mu']['sha256']='0'*64
    else: value['configuration']['averaging']=True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        campaign.report_campaign(root,tmp_path/'bad',path,[1,2,10],shape=SHAPE,vs_model=vs)
    assert not (tmp_path/'bad').exists()


def test_historical_aggregate_receipt_and_retained_qc_supported(tmp_path):
    root=tmp_path/'runs'; manifest,vs=fixture(root)
    campaign.aggregate_campaign(root,tmp_path/'aggregate',[1,2,10],shape=SHAPE,vs_model=vs)
    path=tmp_path/'aggregate/receipt.json'; value=json.loads(path.read_text())
    value.pop('configuration'); path.write_text(json.dumps(value))
    result=campaign.report_campaign(root,tmp_path/'report',path,[1,2,10],shape=SHAPE,campaign_receipt=manifest)
    assert result['QC_basis']=='retained hash-bound execution/aggregate QC records'


def test_actual_cli_families_and_qc_record_binding(tmp_path,capsys):
    root=tmp_path/'runs'; manifest,vs=fixture(root)
    shared=['--runs-root',str(root),'--shots','1,2,10','--shape','3,2','--campaign-receipt',str(manifest),'--vs-model',str(vs)]
    assert campaign.main(['aggregate','--output',str(tmp_path/'agg'),*shared])==0
    qc_path=tmp_path/'qc.json'
    meta=json.loads((root/'shot_001/run_metadata.json').read_text())
    qc_path.write_text(json.dumps({'run':'shot_001','raw_identities':meta['raw_images'],'qc':meta['qc']}))
    assert campaign.main(['report','--output',str(tmp_path/'report'),'--aggregate-receipt',
                          str(tmp_path/'agg/receipt.json'),'--qc-record',str(qc_path),*shared])==0
    result=json.loads((tmp_path/'report/summary.json').read_text())
    assert result['qc_records'][0]['shot']==1
    assert '"solver_executions": 0' in capsys.readouterr().out


def test_raw_water_negative_zero_fails_qc_without_output(tmp_path):
    root=tmp_path/'runs'; _,vs=fixture(root)
    path=root/'shot_001/raw/migration.image_mu_raw.bin'
    values=np.fromfile(path,dtype='<f8').reshape(SHAPE); values[0,0]=-0.
    path.write_bytes(values.tobytes())
    edit_metadata(root,1,lambda m:m['raw_images']['mu'].update(sha256=sha256_file(path)))
    # Summation would hide -0: reject each raw input before accumulation.
    with pytest.raises(ValueError,match=r'exact \+0'):
        campaign.aggregate_campaign(root,tmp_path/'bad',[1,2,10],shape=SHAPE,vs_model=vs)
    assert not (tmp_path/'bad').exists()


def test_production_report_independently_retains_all_scientific_caveats(tmp_path,monkeypatch):
    import subprocess
    def forbidden(*args,**kwargs):
        raise AssertionError('production post-processing must never execute DENISE or a compiler')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    root=tmp_path/'runs'; manifest,vs=fixture(root,ids=(1,68))
    campaign.aggregate_campaign(root,tmp_path/'aggregate',[1,68],shape=SHAPE,campaign_receipt=manifest,vs_model=vs)
    campaign.report_campaign(root,tmp_path/'report',tmp_path/'aggregate/receipt.json',[1,68],
                             shape=SHAPE,campaign_receipt=manifest,vs_model=vs)
    # Requirements come from the task/scientific contract, NOT campaign.CAVEATS.
    for filename in ('summary.json','index.html'):
        text=(tmp_path/'report'/filename).read_text().lower()
        assert 'physical water remains vs=0 / mu=0' in text
        assert 'therefore background mu=0' in text
        assert 'no vs floor, shear floor or epsilon-mu substitution is applied' in text
        assert 'water gmu=+0 is the restricted-parameter-space migration result' in text
        assert 'does not imply an absent fluid wavefield, adjoint or lambda sensitivity' in text
        assert 'no negative-zero sign bits' in text and 'no post-mask' in text
        assert 'shot 68' in text and 'norm/qc observation, not a rejected shot' in text
        assert 'strong first-solid-row contribution' in text and 'later scientific interpretation' in text
        assert '15 hz low-pass exceeds' in text and 'taylor-fd4 dispersion advisory' in text
        assert 'do not establish final broadband migration accuracy' in text
        assert 'no fluid-3/4 acceptance' in text and 'no cuda fluid support' in text
    summary=json.loads((tmp_path/'report/summary.json').read_text())
    assert summary['shot_68_observation']['rejected'] is False
    assert summary['observations_are_not_acceptance_thresholds']
    assert summary['solver_executions']==0
