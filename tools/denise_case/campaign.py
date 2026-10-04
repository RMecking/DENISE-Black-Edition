"""Portable, solver-free raw A2 aggregation and campaign product reporting.

Read historical metadata unchanged. Resolve products from their run-relative
locations, not historical absolute paths. Never write into per-shot directories
or overwrite an existing aggregate/report directory.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from .fluid2 import CORE_SHA, image_checks
from .model_io import read_grid, sha256_file
from .rtm import read_raw_image

CAVEATS = (
    'Physical water remains Vs=0 / mu=0: the water layer has Vs=0 and therefore background mu=0; no Vs floor, shear floor or epsilon-mu substitution is applied.',
    'Water gMu=+0 is the restricted-parameter-space migration result: fluid mu is frozen; zero water gMu does not imply an absent fluid wavefield, adjoint or lambda sensitivity.',
    'Raw water gMu must be exact +0, with no negative-zero sign bits, before display; no post-mask or shear floor.',
    'Shot 68 remains a documented norm/QC observation, not a rejected shot or an amplitude acceptance threshold.',
    'The strong first-solid-row contribution remains preserved for later scientific interpretation.',
    'The preserved 15 Hz low-pass exceeds the frozen Taylor-FD4 dispersion advisory (~5.50625 Hz).',
    'No FLUID-3/4 acceptance and no CUDA fluid support are claimed.',
    'Campaign integration and raw production/QC do not establish final broadband migration accuracy.',
)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')


def file_identity(path, relative_to):
    return {'path':Path(path).relative_to(relative_to).as_posix(),
            'bytes':Path(path).stat().st_size,'sha256':sha256_file(Path(path))}


def shot_ids(value):
    """CLI IDs: comma-separated integers or inclusive ranges; duplicates fail."""
    values=[]
    for part in value.split(','):
        if '-' in part:
            first,last=[int(v) for v in part.split('-')]
            require(first<=last,'descending physical shot range')
            values.extend(range(first,last+1))
        else:
            values.append(int(part))
    return expected_ids(values)


def expected_ids(values):
    values=list(values)
    require(values and all(type(v) is int and v>0 for v in values),'positive physical shot IDs required')
    require(len(values)==len(set(values)),'duplicate expected physical shot IDs')
    return sorted(values)


def shape_contract(shape):
    shape=tuple(shape)
    require(len(shape)==2 and all(type(v) is int and v>0 for v in shape),'positive [depth_y,x] shape required')
    return shape


def discover_campaign(runs_root, shots, *, shape=(174,500), authorized_core_sha=CORE_SHA,
                      campaign_receipt=None, expected_binary_sha256=None):
    """Validate authoritative singleton CPU receipts and their current raw bytes."""
    root=Path(runs_root).resolve(); shape=shape_contract(shape); shots=expected_ids(shots)
    require(len(authorized_core_sha)==40 and all(c in '0123456789abcdef' for c in authorized_core_sha),
            'full authorized core SHA required')
    result={}; binary=expected_binary_sha256
    for path in sorted(root.glob('shot_*/run_metadata.json')):
        meta=load(path)
        physical=meta.get('authoritative_shots')
        require(isinstance(physical,list) and len(physical)==1 and type(physical[0]) is int
                and physical[0]>0,'independent singleton physical-shot receipt required')
        shot=physical[0]
        require(shot not in result,f'duplicate physical shot ID {shot}')
        require(path.parent.name==f'shot_{shot:03d}','physical shot directory/metadata mismatch')
        require(meta.get('complete') is True and meta.get('returncode')==0,f'shot {shot}: incomplete')
        require(meta.get('backend')=='CPU-M9' and meta.get('core_sha')==authorized_core_sha,
                f'shot {shot}: CPU/frozen core identity mismatch')
        current_binary=meta.get('binary_sha256')
        require(isinstance(current_binary,str) and len(current_binary)==64,'recorded solver identity missing')
        binary=binary or current_binary
        require(current_binary==binary,'campaign executable identities differ')
        provenance=path.parent/'provenance.json'
        require(sha256_file(provenance)==meta['provenance_sha256'],'per-shot provenance identity changed')
        raw={}
        for key in ('lambda','mu'):
            record=meta['raw_images'][key]
            require(record.get('shape_y_x')==list(shape) and record.get('dtype')=='<f8'
                    and record.get('order') in ('row-major [y,x]','row-major [depth_y,x]'),
                    f'shot {shot}: wrong raw image datatype/layout/shape')
            target=path.parent/f'raw/migration.image_{key}_raw.bin'
            require(record.get('bytes')==shape[0]*shape[1]*8 and target.stat().st_size==record['bytes'],
                    f'shot {shot}: wrong image size')
            require(sha256_file(target)==record['sha256'],f'shot {shot}: raw {key} hash changed')
            read_raw_image(target,shape[1],shape[0])  # exact FP64 size + fully finite
            raw[key]=file_identity(target,root)
        result[shot]={'shot':shot,'metadata':file_identity(path,root),
                     'provenance':file_identity(provenance,root),'raw':raw,'meta':meta}
    require(set(result)==set(shots),f'missing/unexpected physical shot IDs: expected {shots}, found {sorted(result)}')
    records=[result[s] for s in shots]
    if campaign_receipt is not None:
        campaign=load(campaign_receipt)
        require(set(campaign['per_shot'])=={str(s) for s in shots},'campaign receipt physical shot coverage differs')
        require(campaign.get('completed',len(shots))==len(shots) and campaign.get('failed',0)==0,
                'campaign receipt is incomplete')
        for record in records:
            old=campaign['per_shot'][str(record['shot'])]
            require(old['metadata']['sha256']==record['metadata']['sha256'],'campaign metadata hash differs')
            for key in ('lambda','mu'):
                expected=old.get('raw',{}).get(key) or old[f'raw_{key}']
                require(expected['sha256']==record['raw'][key]['sha256'],'campaign raw input hash differs')
    return records


def arrays(root, record, shape):
    return {key:read_raw_image(Path(root)/record['raw'][key]['path'],shape[1],shape[0])
            for key in ('lambda','mu')}


def unchanged(root, records):
    for record in records:
        for entry in [record['metadata'],record['provenance'],*record['raw'].values()]:
            require(sha256_file(Path(root)/entry['path'])==entry['sha256'],'input changed during post-processing')


def fresh_output(output, root, records):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('refusing to overwrite existing aggregate/report output')
    for record in records:
        run=(Path(root)/record['metadata']['path']).parent.resolve()
        require(not output.is_relative_to(run) and not run.is_relative_to(output),
                'output must be separate from per-shot raw/derived/input directories')
    return output


def write_new(path, payload):
    with Path(path).open('xb') as stream:
        stream.write(payload)


def configuration(shape, core, binary):
    return {'shape_y_x':list(shape),'dtype':'<f8','order':'row-major [depth_y,x]',
            'accumulation_dtype':'<f8','authorized_core_sha':core,'binary_sha256':binary,
            'operation':'ascending physical-shot unweighted FP64 sum',
            'averaging':False,'normalization':False,'smoothing':False,'mute':False,
            'illumination_weighting':False,'preconditioning':False,'parameter_transform':False}


def aggregate_campaign(runs_root, output, shots=range(1,101), *, shape=(174,500),
                       authorized_core_sha=CORE_SHA, campaign_receipt=None, vs_model=None):
    """Reproduce the accepted raw sum; never run DENISE or alter inputs."""
    shape=shape_contract(shape); root=Path(runs_root).resolve()
    records=discover_campaign(root,shots,shape=shape,authorized_core_sha=authorized_core_sha,
                              campaign_receipt=campaign_receipt)
    output=fresh_output(output,root,records)
    vs=None if vs_model is None else read_grid(Path(vs_model),shape[1],shape[0])
    sums={key:np.zeros(shape,dtype='<f8') for key in ('lambda','mu')}
    for record in records:
        raw=arrays(root,record,shape)
        if vs is not None:
            image_checks(raw['lambda'],raw['mu'],vs)
        for key,values in raw.items():
            np.add(sums[key],values,out=sums[key])
    require(all(np.isfinite(a).all() for a in sums.values()),'aggregate contains nonfinite values')
    qc=None
    if vs_model is not None:
        qc=image_checks(sums['lambda'],sums['mu'],vs)
    unchanged(root,records)
    output.mkdir(parents=True,exist_ok=False); (output/'raw').mkdir()
    outputs={}
    for key,values in sums.items():
        path=output/f'raw/migration.image_{key}_raw.bin'
        write_new(path,values.astype('<f8').tobytes(order='C'))
        outputs[key]=file_identity(path,output)
    receipt={'schema_version':1,'order':[r['shot'] for r in records],
             'operation':'ascending unweighted FP64 sum, no normalization/conditioning',
             'configuration':configuration(shape,authorized_core_sha,records[0]['meta']['binary_sha256']),
             'input_hashes':{str(r['shot']):{k:v['sha256'] for k,v in r['raw'].items()} for r in records},
             'per_shot_inputs':[{k:r[k] for k in ('shot','metadata','provenance','raw')} for r in records],
             'outputs':outputs,'qc':qc,'solver_executions':0,'raw_vs_derived_separate':True}
    if campaign_receipt is not None:
        receipt['campaign_receipt_sha256']=sha256_file(Path(campaign_receipt))
    if vs_model is not None:
        receipt['vs_model_sha256']=sha256_file(Path(vs_model))
    unchanged(root,records)
    write_new(output/'receipt.json',encoded(receipt))
    return receipt


def verify_aggregate(receipt_path, records, shape):
    receipt_path=Path(receipt_path); receipt=load(receipt_path)
    require(receipt['order']==[r['shot'] for r in records],'aggregate accumulation order differs')
    require(receipt['input_hashes']=={str(r['shot']):{k:v['sha256'] for k,v in r['raw'].items()} for r in records},
            'aggregate input identities differ')
    if 'configuration' in receipt:
        require(receipt['configuration']==configuration(shape,records[0]['meta']['core_sha'],
                records[0]['meta']['binary_sha256']),'aggregate operation/layout configuration differs')
    else:
        require(receipt['operation']=='ascending unweighted FP64 sum, no normalization/conditioning',
                'historical aggregate operation differs')
    result={}
    for key in ('lambda','mu'):
        path=receipt_path.parent/f'raw/migration.image_{key}_raw.bin'
        require(path.stat().st_size==shape[0]*shape[1]*8,'aggregate raw size differs')
        require(sha256_file(path)==receipt['outputs'][key]['sha256'],'aggregate output hash differs')
        read_raw_image(path,shape[1],shape[0])
        result[key]=file_identity(path,receipt_path.parent)
    return receipt,result


def distribution(values, records):
    values=np.asarray(values,dtype=np.float64)
    require(np.isfinite(values).all(),'nonfinite reported metric')
    return {'min':float(values.min()),'median':float(np.median(values)),'max':float(values.max()),
            'min_shot':records[int(np.argmin(values))]['shot'],'max_shot':records[int(np.argmax(values))]['shot']}


def report_campaign(runs_root, output, aggregate_receipt, shots=range(1,101), *,
                    shape=(174,500), authorized_core_sha=CORE_SHA, campaign_receipt=None,
                    vs_model=None, qc_records=()):
    """Validate existing products and produce deterministic JSON/HTML, no solver."""
    shape=shape_contract(shape); root=Path(runs_root).resolve()
    records=discover_campaign(root,shots,shape=shape,authorized_core_sha=authorized_core_sha,
                              campaign_receipt=campaign_receipt)
    output=fresh_output(output,root,records)
    aggregate_receipt=Path(aggregate_receipt).resolve()
    old,aggregate_outputs=verify_aggregate(aggregate_receipt,records,shape)
    # Reports may not be written underneath an existing aggregate's raw/derived area.
    require(not output.is_relative_to(aggregate_receipt.parent),'report output must be separate from aggregate')
    vs=None if vs_model is None else read_grid(Path(vs_model),shape[1],shape[0])
    qcs=[]
    for record in records:
        if vs is None:
            qc=record['meta'].get('qc')
            require(qc and qc.get('water_mu_exact_positive_zero') and qc['water_mu']['max_abs']==0,
                    'physical-water QC records or explicit source Vs model required')
        else:
            raw=arrays(root,record,shape)
            qc=image_checks(raw['lambda'],raw['mu'],vs)
        qcs.append(qc)
    aggregate_qc=old.get('qc')
    if vs is not None:
        raw={k:read_raw_image(aggregate_receipt.parent/v['path'],shape[1],shape[0]) for k,v in aggregate_outputs.items()}
        aggregate_qc=image_checks(raw['lambda'],raw['mu'],vs)
    require(aggregate_qc and aggregate_qc.get('water_mu_exact_positive_zero')
            and aggregate_qc['water_mu']['max_abs']==0,'aggregate physical-water QC required')
    external_qc=[]
    by_shot={r['shot']:r for r in records}
    for path in qc_records:
        value=load(path); shot=int(value['run'].removeprefix('shot_'))
        require(shot in by_shot,'QC record is outside campaign')
        require(all(value['raw_identities'][k]['sha256']==by_shot[shot]['raw'][k]['sha256']
                    for k in ('lambda','mu')),'QC record raw identities differ')
        external_qc.append({'shot':shot,'sha256':sha256_file(Path(path)),'qc':value['qc']})
    ranges={name:distribution([qc[name]['l2'] for qc in qcs],records)
            for name in ('water_lambda','water_interior_lambda','solid_lambda','solid_mu')}
    ranges['runtime_seconds']=distribution([r['meta']['runtime_seconds'] for r in records],records)
    result={'schema_version':1,'scope':'solver-free campaign product/QC report; not a new scientific acceptance',
            'authorized_execution_core':authorized_core_sha,'shots':[r['shot'] for r in records],
            'completed':len(records),'configuration':configuration(shape,authorized_core_sha,records[0]['meta']['binary_sha256']),
            'per_shot':[{**{k:r[k] for k in ('shot','metadata','provenance','raw')},'qc':qc}
                        for r,qc in zip(records,qcs)],
            'aggregate':{'receipt_sha256':sha256_file(aggregate_receipt),'outputs':aggregate_outputs,'qc':aggregate_qc},
            'qc_records':sorted(external_qc,key=lambda r:(r['shot'],r['sha256'])),
            'distributions':ranges,'total_recorded_cpu_seconds':sum(r['meta']['runtime_seconds'] for r in records),
            'water_mu_max_abs':0.0,'water_mu_exact_positive_zero':True,
            'QC_basis':'recomputed using source Vs' if vs is not None else 'retained hash-bound execution/aggregate QC records',
            'scientific_caveats':list(CAVEATS),'solver_executions':0,'raw_vs_derived_separate':True,
            'observations_are_not_acceptance_thresholds':True}
    if 68 in by_shot:
        result['shot_68_observation']={'qc':qcs[result['shots'].index(68)],'rejected':False}
    if campaign_receipt is not None:
        result['campaign_receipt_sha256']=sha256_file(Path(campaign_receipt))
    if vs_model is not None:
        result['vs_model_sha256']=sha256_file(Path(vs_model))
    unchanged(root,records)
    verify_aggregate(aggregate_receipt,records,shape)
    output.mkdir(parents=True,exist_ok=False)
    write_new(output/'summary.json',encoded(result))
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Marmousi-II campaign product report</title><style>body{max-width:1150px;margin:2em auto;padding:0 1em;font:16px system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Marmousi-II campaign product/QC report</h1><p>Existing products only; no DENISE execution. Raw lambda/mu JT images are not display-processed products.</p><ul>'
    page+=''.join('<li>'+html.escape(text)+'</li>' for text in CAVEATS)
    page+='</ul><p><a href="summary.json">Machine-readable summary</a></p><pre>'+html.escape(encoded(result).decode())+'</pre></html>'
    write_new(output/'index.html',page.encode('utf-8'))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('aggregate','report'))
    parser.add_argument('--runs-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shots',type=shot_ids,default=list(range(1,101)))
    parser.add_argument('--shape',default='174,500',help='depth_y,x')
    parser.add_argument('--authorized-core-sha',default=CORE_SHA)
    parser.add_argument('--campaign-receipt',type=Path)
    parser.add_argument('--aggregate-receipt',type=Path)
    parser.add_argument('--vs-model',type=Path)
    parser.add_argument('--qc-record',type=Path,action='append',default=[])
    args=parser.parse_args(argv)
    shape=shape_contract([int(v) for v in args.shape.split(',')])
    kwargs={'shape':shape,'authorized_core_sha':args.authorized_core_sha,
            'campaign_receipt':args.campaign_receipt,'vs_model':args.vs_model}
    if args.action=='aggregate':
        result=aggregate_campaign(args.runs_root,args.output,args.shots,**kwargs)
    else:
        if args.aggregate_receipt is None:
            parser.error('report requires --aggregate-receipt')
        result=report_campaign(args.runs_root,args.output,args.aggregate_receipt,args.shots,
                               qc_records=args.qc_record,**kwargs)
    print(json.dumps({'action':args.action,'shots':result.get('order',result.get('shots')),
                      'solver_executions':0},sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
