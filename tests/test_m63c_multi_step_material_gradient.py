"""M6.3c-7c-b2 integrated reverse-time material-gradient verification."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import pytest

CASES=(
 ("n1",1,1,1,1,0,0,"all"),("n2",1,1,2,3,1,0,"all"),
 ("n5_first",2,1,5,1,0,0,"first"),("n5_middle",1,2,5,3,1,0,"middle"),
 ("n5_last",2,2,5,1,1,0,"last"),("n5_cancel",1,1,5,3,0,0,"cancel"),
 ("mpi_cpml",2,1,5,1,1,2,"all"),("mpi_fs",1,2,5,3,0,2,"all"),
 ("corner",2,2,5,1,1,2,"all"),
 ("rho_only",1,1,5,1,0,0,"rho_only"),
 ("x_only",1,1,5,3,1,0,"x_only"),
 ("y_only",1,1,5,1,1,0,"y_only"),
)

@pytest.fixture(scope="module")
def harness(tmp_path_factory,repository_root:Path):
 c=shutil.which("mpicc");m=shutil.which("mpiexec") or shutil.which("mpirun");assert c and m
 exe=tmp_path_factory.mktemp("m63c7cb2")/"m63c7cb2"
 sources=["tests/utilities/m63c_multi_step_material_gradient_harness.c","src/SH/update_v_PML_SH.c","src/SH/update_s_visc_PML_SH.c","src/SH/exchange_v_SH.c","src/SH/exchange_s_SH.c","src/SH/surface_elastic_SH.c","src/SH/visco_sh_gsls_vjp.c","src/SH/visco_sh_material_vjp.c","src/SH/visco_sh_material_timestep_vjp.c","src/SH/visco_sh_material_observable.c","src/SH/update_s_visc_PML_SH_adjoint.c","src/SH/update_v_PML_SH_adjoint.c","src/SH/exchange_v_SH_adjoint.c","src/SH/exchange_s_SH_adjoint.c","src/SH/surface_elastic_SH_adjoint.c","src/SH/visco_sh_full_state_adjoint_step.c","src/SH/visco_sh_reverse_time_adjoint.c","src/SH/visco_sh_reverse_time_material_gradient.c","src/SH/visco_sh_material_gradient_assembly.c","src/SH/matcopy_SH.c","src/SH/matcopy_SH_adjoint.c","src/SH/av_mu_SH.c","src/av_tau.c","src/q_parameterization.c"]
 cmd=[c,"-std=c99","-O2","-fcommon","-I",str(repository_root/"include"),*(str(repository_root/s) for s in sources),"-o",str(exe),"-lm"]
 r=subprocess.run(cmd,cwd=repository_root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);assert r.returncode==0,r.stdout
 return m,exe

@pytest.mark.parametrize("case",CASES,ids=lambda x:x[0])
def test_integrated_driver_matches_locked_explicit_composition(harness,tmp_path,case):
 name,npx,npy,nsteps,invmat,qmode,fw,mode=case;launcher,exe=harness;out=tmp_path/name;out.mkdir()
 cmd=[launcher,"--oversubscribe","-n",str(npx*npy),str(exe),str(npx),str(npy),str(nsteps),str(invmat),str(qmode),"1" if name=="mpi_fs" else "0",str(fw),mode,str(out)]
 r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90);assert r.returncode==0,r.stdout
 rec=[json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")];assert len(rec)==1,r.stdout;d=rec[0];print("M63C7CB2 "+json.dumps({"case":name,**d},sort_keys=True));assert d["state_error"]==0.0;assert d["signal_error"]==0.0;assert d["gradient_error"]<=2e-6;assert d["input_change"]==0.0

@pytest.mark.parametrize("mode",("invalid_nsteps","invalid_nx","invalid_ny","invalid_dtinv","invalid_missing","invalid_context"))
def test_preflight_rejects_before_state_or_gradient_mutation(harness,tmp_path,mode):
 launcher,exe=harness;out=tmp_path/mode;out.mkdir();cmd=[launcher,"--oversubscribe","-n","1",str(exe),"1","1","2","1","0","0","0",mode,str(out)]
 r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60);assert r.returncode==0,r.stdout;rec=[json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")];assert len(rec)==1,r.stdout;assert rec[0]["preflight_status"]==-1;assert rec[0]["preflight_change"]==0.0

def test_repeated_invocation_is_deterministic(harness,tmp_path):
 launcher,exe=harness;checks=[]
 for repeat in range(2):
  out=tmp_path/f"repeat-{repeat}";out.mkdir()
  cmd=[launcher,"--oversubscribe","-n","1",str(exe),"1","1","5","1","1","0","0","all",str(out)]
  r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60);assert r.returncode==0,r.stdout
  rec=[json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")];assert len(rec)==1,r.stdout;checks.append(rec[0]["gradient_checksum"])
 assert checks[0]==checks[1]

def test_driver_contract_is_separate_streaming_and_dtinv1(repository_root:Path):
 s=(repository_root/"src/SH/visco_sh_reverse_time_material_gradient.c").read_text();start=s.index("int visco_sh_reverse_time_adjoint_material(");body=s[start:]
 assert "trajectory->steps[n]" in body
 assert "visco_sh_full_state_adjoint_step_material(" in body
 assert body.count("visco_sh_temporal_native_gradient_accumulate(")==1
 assert body.count("visco_sh_distributed_material_gradient_vjp(")==1
 assert "trajectory->dtinv != 1" in s
 fixed=(repository_root/"src/SH/visco_sh_reverse_time_adjoint.c").read_text()
 assert "visco_sh_reverse_time_adjoint(" in fixed
 assert "visco_sh_reverse_time_adjoint_material(" not in fixed
 for forbidden in ("grad_obj_sh","FWI_SH","DTINV"):
  assert forbidden not in body

def test_locked_and_active_paths_remain_untouched(repository_root:Path):
 paths=["src/SH/visco_sh_full_state_adjoint_step.c","src/SH/visco_sh_material_timestep_vjp.c","src/SH/matcopy_SH_adjoint.c","src/SH/visco_sh_material_observable.c","src/SH/FWI_SH.c","src/SH/FWI_SH_visc.c","src/SH/grad_obj_sh.c","src/SH/grad_obj_sh_visc.c"]
 r=subprocess.run(["git","diff","--name-only","f67daaff71f98b1f7ef048821175b56e9ea73ac8","--",*paths],cwd=repository_root,text=True,stdout=subprocess.PIPE,check=True);assert r.stdout==""
