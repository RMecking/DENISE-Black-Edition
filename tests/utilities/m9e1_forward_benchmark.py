"""Reproducible warm-runtime end-to-end M9e-1 verification benchmark.

Includes create, canonical maps, resident forward, all output downloads, destroy.
This is a verification harness, not a MODE=2 backend or performance dispatch.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tests.physics.test_m9e1_cuda_elastic_psv_forward import (
    Config,Options,Diagnostics,F,P,fp,fixture,outputs,check,cpu_run,create,download,same,metrics,
)

def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu',required=True);p.add_argument('--nofma',required=True)
    p.add_argument('--fma',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    cpu=C.CDLL(args.cpu);cpu.m9e1_cpu_view.argtypes=[C.POINTER(Config),F,F,F,F,F]
    libs={}
    for name,path in [('nofma',args.nofma),('fma',args.fma)]:
        l=C.CDLL(path);l.denise_cuda_m9_last_error.restype=C.c_char_p
        l.denise_cuda_m9_create.argtypes=[C.POINTER(Config),C.POINTER(Options),C.POINTER(P)]
        l.denise_cuda_m9_prepare.argtypes=[P];l.denise_cuda_m9_download.argtypes=[P,F,F,F,F]
        l.denise_cuda_m9_destroy.argtypes=[C.POINTER(P)];libs[name]=l
    cfg,a=fixture('heterogeneous',nx=97,ny=79,nt=120)
    def run(name):
        start=time.perf_counter()
        if name=='cpu': out=cpu_run(cpu,cfg)
        else:
            lib=libs[name];ctx=create(lib,cfg)
            try:
                check(lib,lib.denise_cuda_m9_prepare(ctx));out=download(lib,ctx,cfg)
            finally: check(lib,lib.denise_cuda_m9_destroy(C.byref(ctx)))
        return time.perf_counter()-start,out
    ref=run('cpu')[1]
    for name in libs:run(name)  # Warm runtime and caches, report it explicitly.
    timings={n:[] for n in ['cpu',*libs]}
    for _ in range(7):
        for name in timings:
            elapsed,out=run(name);timings[name].append(elapsed)
            if name!='fma':assert all(same(x,y) for x,y in zip(out,ref))
            else:
                for x,y in zip(out,ref):
                    z=metrics(x,y);assert z['rel_l2']<=2e-6 and z['peak_normalized']<=8e-6,z
    result={'fixture':{'nx':cfg.nx,'ny':cfg.ny,'nt':cfg.nt,'receivers':cfg.receiver_count},
        'scope':'warm runtime; complete context create + prepare + all 18 arrays download + destroy',
        'seconds':timings,'median_seconds':{n:statistics.median(t) for n,t in timings.items()}}
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
