"""Test-only adapters: frozen zero-shear maps + independent M9 FP64 waves.

No production products enter expected values. Original reference files remain
unchanged. Each isolated module receives only the frozen material-map functions;
the already checked M9 state/CPML/surface tangent and reverse graphs are reused.
"""
from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from tests.utilities import zero_shear_reference as z
from tests.utilities import elastic_psv_born_reference as b


def fixture(pattern="horizontal", fs=0, fw=0, nx=24, ny=20, nt=120):
    y, x = np.indices((ny, nx))
    mu = np.full((ny, nx), 3.375)
    if pattern == "homogeneous":
        mu[:] = 0
    elif pattern == "horizontal":
        mu[:ny//2+1] = 0
    elif pattern == "offset":
        mu[y < ny//2 + (x >= nx//2)] = 0
    elif pattern == "vertical":
        mu[:, :nx//2+1] = 0
    elif pattern != "solid":
        raise ValueError(pattern)
    rho = np.where(mu == 0, 1., 1.5).astype(np.float32).astype(float)
    lam = np.where(mu == 0, 4., 6.75).astype(np.float32).astype(float)
    mu = mu.astype(np.float32).astype(float)
    return replace(b.make_experiment(), name="fluid2", nx=nx, ny=ny, dh=1.,
        dt=.1, nt=nt, fw=fw, cpml=bool(fw), rho=rho, lam=lam, mu=mu,
        sources=((nx//2+1, ny//2+1),),
        receivers=((nx//2+4,ny//2+1),(nx//2+1,ny//2+4),
                   (nx//2-1,ny//2-1),(2,2)),
        fc=.5, source_t0=3., source_amplitude=1., pml_fpml=.1)


def isolated(name):
    path = Path(b.__file__).with_name(name + ".py")
    key = "_fluid2_" + name
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def maps(mu):
    return z.material(np.ones_like(mu), np.full_like(mu, 4.), mu)


def module(fs, source):
    r = isolated("elastic_psv_free_surface_reference" if fs else "elastic_psv_born_reference")
    r._source_samples = lambda exp: source
    if fs:
        r.corner = lambda mu: maps(mu).corner
        r.corner_jvp = lambda mu, dm: z.corner_jvp(maps(mu), dm)
        r.corner_vjp = lambda mu, bar: z.corner_vjp(maps(mu), bar)
        def coefficients(lam, mu):
            _, alpha, a, al, bl = z.surface(lam,mu,np.ones_like(mu),np.zeros_like(mu))
            _, _, _, am, bm = z.surface(lam,mu,np.zeros_like(mu),np.ones_like(mu))
            return alpha,a,al,am,bl,bm
        r.surface_coefficients = coefficients
    else:
        r.shear_corner_mu = lambda mu: maps(mu).corner
        r.shear_corner_tangent = lambda mu, dm: z.corner_jvp(maps(mu), dm)
        r.shear_corner_transpose = lambda mu, bar: z.corner_vjp(maps(mu), bar)
    return r


def trajectory(exp, source, fs=0):
    """FP64 forward operands from frozen independent zero-shear equations."""
    r = module(fs, source)
    if fs:
        data, tapes, _ = r.forward(exp)
        return r, tapes, data
    m = z.material(exp.rho, exp.lam, exp.mu)
    state = np.zeros((5, exp.ny, exp.nx))
    memory = np.zeros((8, exp.ny, exp.nx))
    profiles = r.cpml_profiles(exp)
    names = ("xh","y","x","yh","x","xh","yh","y")
    strains = np.zeros((exp.nt,4,exp.ny,exp.nx))
    data = np.zeros((exp.nt,len(exp.receivers),2))
    def corrected(raw, k):
        q, memory[k] = r._pml_forward(exp.dt/exp.dh*raw,memory[k],profiles[names[k]])
        return q
    vx,vy,xx,yy,xy = state
    sj,si = exp.sources[0][1]-1,exp.sources[0][0]-1
    for n, load in enumerate(source):
        vx += m.rx*(corrected(z.derivative(xx,1,True),0)+corrected(z.derivative(xy,0,False),1))
        vy += m.ry*(corrected(z.derivative(xy,1,False),2)+corrected(z.derivative(yy,0,True),3))
        for k,(i,j) in enumerate(exp.receivers):
            data[n,k] = vx[j-1,i-1],vy[j-1,i-1]
        ex = corrected(z.derivative(vx,1,False),4)
        yx = corrected(z.derivative(vy,1,True),5)
        x_y = corrected(z.derivative(vx,0,True),6)
        ey = corrected(z.derivative(vy,0,False),7)
        strains[n] = ex,yx,x_y,ey
        xx += m.lam*(ex+ey)+2*m.mu*ex
        yy += m.lam*(ex+ey)+2*m.mu*ey
        xy += m.corner*(yx+x_y)
        xx[sj,si] += load
        yy[sj,si] += load
    return r,r.BackgroundRun(data,strains,0.,0.,0.),data


def products(exp, source, dl, dm, data, fs=0, prepared=None):
    z.require_direction(z.material(exp.rho,exp.lam,exp.mu),dm)
    r,tapes,_ = trajectory(exp,source,fs) if prepared is None else prepared
    if fs:
        j = r.born(exp,tapes,dl,dm)
        gl,gm = r.adjoint(exp,tapes,data)
    else:
        j = r.born_forward(exp,tapes,dl,dm)
        image = r.born_adjoint(exp,tapes,data)
        gl,gm = image.image_lambda_raw,image.image_mu_raw
    gm[exp.mu == 0] = 0.
    lhs = math.fsum(float(x)*float(y) for x,y in zip(j.ravel(),data.ravel()))
    rhs = math.fsum(float(x)*float(y) for x,y in zip(dl.ravel(),gl.ravel()))
    rhs += math.fsum(float(x)*float(y) for x,y in zip(dm.ravel(),gm.ravel()))
    scale = max(np.linalg.norm(j)*np.linalg.norm(data),
                np.hypot(np.linalg.norm(dl),np.linalg.norm(dm))*np.hypot(np.linalg.norm(gl),np.linalg.norm(gm)))
    dot = dict(absolute_residual=abs(lhs-rhs),absolute_ceiling=5e-13*scale)
    assert dot["absolute_residual"] <= dot["absolute_ceiling"], dot
    return j,gl,gm,dot


def record(kind, **values):
    path = os.environ.get("DENISE_FLUID2_EVIDENCE")
    if path:
        with open(path,"a") as f:
            f.write(json.dumps(dict(kind=kind,**values))+"\n")
