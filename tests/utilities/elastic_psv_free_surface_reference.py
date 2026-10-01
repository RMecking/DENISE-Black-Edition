"""Independent FP64 flat-surface FD4 elastic P/SV graph and matrix oracle.

No production C is called or parsed. Indices are zero-based physical (y,x).
Ghosts are algebraic, never checkpoint state. See the companion derivation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tests.utilities.elastic_psv_born_reference import (
    Experiment, PMLProfile, _cpml_1d, _source_samples,
)

FD4 = (9.0 / 8.0, -1.0 / 24.0)
EXTRAPOLATE_SURFACE = np.array([35., -35., 21., -5.]) / 16.
VX, VY, SXX, SYY, SXY = range(5)
PSXX, PSXYY, PSXYX, PSYY, PVXX, PVYX, PVXY, PVYY = range(5, 13)
STATE_NAMES = ("vx", "vy", "sxx", "syy", "sxy", "sxx_x", "sxy_y",
               "sxy_x", "syy_y", "vxx", "vyx", "vxy", "vyy")
MEMORY_PROFILES = ("xh", "y", "x", "yh", "x", "xh", "yh", "y")
TERMS = {
    "xb": ((0, 9/8), (-1, -9/8), (1, -1/24), (-2, 1/24)),
    "xf": ((1, 9/8), (0, -9/8), (2, -1/24), (-1, 1/24)),
    "yb": ((0, 9/8), (-1, -9/8), (1, -1/24), (-2, 1/24)),
    "yf": ((1, 9/8), (0, -9/8), (2, -1/24), (-1, 1/24)),
}


def surface_coefficients(lam, mu):
    """Normal strain ratio, plane-stress stiffness, and analytic derivatives."""
    den = lam + 2 * mu
    if np.any(mu <= 0) or np.any(den <= 0):
        raise ValueError("positive mu and lambda+2mu required")
    alpha = lam / den
    a = 4 * mu * (lam + mu) / den
    return alpha, a, 2*mu/den**2, -2*lam/den**2, 4*mu**2/den**2, 4*(lam**2+2*lam*mu+2*mu**2)/den**2


def fixture(*, nx=8, ny=8, nt=32, cpml=True, source=(4, 2)):
    if nx < 4 or ny < 4:
        raise ValueError("FD4 surface fixture requires NX,NY >= 4")
    y, x = np.mgrid[:ny, :nx]
    rho = np.full((ny, nx), 2000., dtype=np.float64)
    lam = 6.44e9 * (1 + .04*np.sin(.7*x+.2*y))
    mu = 5.78e9 * (1 + .03*np.cos(.3*x-.5*y))
    receivers = ((1, 1), (nx//2, 2), (nx//2+1, 1), (nx, 2), (2, ny-1))
    return Experiment("free_surface", nx, ny, 10., .0004, nt, 2 if cpml else 0,
                      cpml, rho, lam, mu, (source,), receivers,
                      source_t0=.003, source_amplitude=1.e6)


def validate_geometry(exp, topology=(1, 1)):
    px, py = topology
    if px < 1 or py < 1 or exp.nx % px or exp.ny % py:
        raise ValueError("positive topology and even subdomains required")
    if exp.nx//px < 2 or exp.ny//py < 4:
        # Uniform domains imply the top-rank requirement governs all ranks.
        raise ValueError("top local_ny >= 4; interior >= 2; local_nx >= 2")
    for i, j in exp.sources:
        if not (1 <= i <= exp.nx and 2 <= j <= exp.ny):
            raise ValueError("explosive source requires physical row j >= 2")
    for i, j in exp.receivers:
        if not (1 <= i <= exp.nx and 1 <= j <= exp.ny):
            raise ValueError("receiver must be a physical velocity node")
    if exp.fw and (2*exp.fw >= exp.nx or exp.fw >= exp.ny-3):
        raise ValueError("CPML layers may not overlap each other or top closure")


def profiles(exp):
    """Fixed base-material CPML coefficients: x edges and bottom only."""
    vmax = float(np.sqrt(np.max((exp.lam+2*exp.mu)/exp.rho)))
    args = dict(damping_speed=vmax, reflection=exp.pml_reflection,
                power=exp.pml_power, kmax=exp.pml_kmax, fpml=exp.pml_fpml)
    out = {}
    for axis, n in (("x", exp.nx), ("y", exp.ny)):
        for half in (False, True):
            p = _cpml_1d(n, exp.dh, exp.dt, exp.fw if exp.cpml else 0,
                         half=half, **args)
            if axis == "y":
                # No top absorber. Preserve only the bottom profile samples.
                active = np.arange(n) >= n-1-exp.fw
                p = PMLProfile(np.where(active,p.kappa,1.),
                               np.where(active,p.a,0.), np.where(active,p.b,1.))
            shape = (1,n) if axis == "x" else (n,1)
            out[axis + ("h" if half else "")] = PMLProfile(*(v.reshape(shape) for v in p))
    return out


def dx(a, kind):
    out = np.zeros_like(a)
    for offset, weight in TERMS[kind]:
        out += weight * np.roll(a, -offset, axis=1)
    return out


def dx_t(a, kind):
    out = np.zeros_like(a)
    for offset, weight in TERMS[kind]:
        out += weight * np.roll(a, offset, axis=1)
    return out


def extend(a, kind, *, alpha=None, qxx=None, qyx=None, h_over_dt=1.):
    """Physical + two ghosts each side; bottom retains M9 periodic closure."""
    ny, nx = a.shape
    out = np.empty((ny+4, nx))
    out[2:ny+2] = a
    if kind == "syy":
        out[2] = 0.
    for m in (1, 2):
        if kind == "syy":
            out[2-m] = -a[m]
        elif kind == "sxy":
            out[2-m] = -a[m-1]
        elif kind == "vx":
            slope = np.einsum("j,ji->i", EXTRAPOLATE_SURFACE, qyx[:4])
            out[2-m] = a[m] + 2*m*h_over_dt*slope
        elif kind == "vy":
            out[2-m] = a[m-1] + (2*m-1)*h_over_dt*alpha*qxx[0]
        else:
            raise ValueError(kind)
        out[ny+1+m] = a[m-1]
    return out


def extend_t(bar, kind, *, alpha=None, qxx=None, qyx=None, h_over_dt=1.):
    """Independent ADD/clear reverse of the explicit extension assignments."""
    ny = bar.shape[0]-4
    out = bar[2:ny+2].copy()
    qxbar = np.zeros_like(out)
    qybar = np.zeros_like(out)
    alphabar = np.zeros(out.shape[1])
    if kind == "syy":
        out[0] = 0.
    for m in (1, 2):
        b = bar[2-m]
        if kind == "syy":
            out[m] -= b
        elif kind == "sxy":
            out[m-1] -= b
        elif kind == "vx":
            out[m] += b
            qybar[:4] += EXTRAPOLATE_SURFACE[:,None] * (2*m*h_over_dt*b)
        elif kind == "vy":
            out[m-1] += b
            qxbar[0] += (2*m-1)*h_over_dt*alpha*b
            if qxx is not None:
                alphabar += (2*m-1)*h_over_dt*qxx[0]*b
        else:
            raise ValueError(kind)
        out[m-1] += bar[ny+1+m]
    return out, qxbar, qybar, alphabar


def dy_extended(a, kind):
    ny = a.shape[0]-4
    return sum(w*a[2+o:2+o+ny] for o,w in TERMS[kind])


def dy_extended_t(a, kind):
    out = np.zeros((a.shape[0]+4, a.shape[1]))
    for o,w in TERMS[kind]:
        out[2+o:2+o+a.shape[0]] += w*a
    return out


def pml(q, old, p):
    new = p.b*old+p.a*q
    return q/p.kappa+new, new


def pml_t(qbar, newbar, p):
    total = qbar+newbar
    return qbar/p.kappa+p.a*total, p.b*total


def corner(mu):
    return 4/sum(1/np.roll(mu,(-j,-i),(0,1)) for j,i in ((0,0),(0,1),(1,0),(1,1)))


def corner_jvp(mu, dm):
    c = corner(mu)
    return c*c/4 * sum(np.roll(dm,(-j,-i),(0,1))/np.roll(mu,(-j,-i),(0,1))**2
                       for j,i in ((0,0),(0,1),(1,0),(1,1)))


def corner_vjp(mu, bar):
    common = corner(mu)**2/4*bar
    return sum(np.roll(common/np.roll(mu,(-j,-i),(0,1))**2,(j,i),(0,1))
               for j,i in ((0,0),(0,1),(1,0),(1,1)))


@dataclass
class Tape:
    strain: tuple
    raw_y: tuple
    sample: np.ndarray


class SurfaceOperator:
    """Full source-free timestep; material tangent and strict reverse graph."""
    def __init__(self, exp, *, lam=None, mu=None, topology=(1,1)):
        validate_geometry(exp,topology)
        self.exp = exp
        self.topology = topology
        self.lam = exp.lam if lam is None else lam
        self.mu = exp.mu if mu is None else mu
        self.p = profiles(exp)  # Frozen coefficient operator, also for nonlinear FD.
        self.coeff = exp.dt/exp.dh
        self.alpha, self.a, self.al, self.am, self.bl, self.bm = surface_coefficients(self.lam[0],self.mu[0])
        self.mc = corner(self.mu)
        self.rx = 2/(exp.rho+np.roll(exp.rho,-1,1))
        self.ry = 2/(exp.rho+np.roll(exp.rho,-1,0))

    def zero(self):
        return np.zeros((13,self.exp.ny,self.exp.nx))

    def x(self,a,kind):
        return distributed_derivative(a,kind,self.topology)

    def y(self,a,kind,field,**kwargs):
        return distributed_derivative(a,kind,self.topology,field=field,**kwargs)

    def xt(self,a,kind):
        return distributed_derivative_t(a,kind,self.topology)[0]

    def yt(self,a,kind,field,**kwargs):
        return distributed_derivative_t(a,kind,self.topology,field=field,**kwargs)

    def step(self, state):
        z = state.copy()
        z[SYY,0] = 0.
        c = self.coeff
        raw = (c*self.x(z[SXX],"xf"), c*self.y(z[SXY],"yb","sxy"),
               c*self.x(z[SXY],"xb"), c*self.y(z[SYY],"yf","syy"))
        q = []
        for k,r,pname in zip((PSXX,PSXYY,PSXYX,PSYY),raw,("xh","y","x","yh")):
            corr,z[k] = pml(r,z[k],self.p[pname]); q.append(corr)
        z[VX] += self.rx*(q[0]+q[1])
        z[VY] += self.ry*(q[2]+q[3])
        sample = np.array([[z[VX,j-1,i-1],z[VY,j-1,i-1]] for i,j in self.exp.receivers])
        qxx,z[PVXX] = pml(c*self.x(z[VX],"xb"),z[PVXX],self.p["x"])
        qyx,z[PVYX] = pml(c*self.x(z[VY],"xf"),z[PVYX],self.p["xh"])
        rawxy = c*self.y(z[VX],"yf","vx",qyx=qyx,h_over_dt=1/c)
        rawyy = c*self.y(z[VY],"yb","vy",alpha=self.alpha,qxx=qxx,h_over_dt=1/c)
        qxy,z[PVXY] = pml(rawxy,z[PVXY],self.p["yh"])
        qyy,z[PVYY] = pml(rawyy,z[PVYY],self.p["y"])
        div = qxx+qyy
        z[SXX] += self.lam*div+2*self.mu*qxx
        # Replace the surface increment, not the persistent old sxx.
        z[SXX,0] += self.a*qxx[0] - (self.lam[0]*div[0]+2*self.mu[0]*qxx[0])
        z[SYY] += self.lam*div+2*self.mu*qyy
        z[SYY,0] = 0.
        z[SXY] += self.mc*(qyx+qxy)
        return z, Tape((qxx,qyx,qxy,qyy),(rawxy,rawyy),sample)

    def tangent(self, dstate, tape, dl, dm):
        # The homogeneous state graph equals the fixed-material step.
        out, dtape = self.step(dstate)
        qxx,qyx,qxy,qyy = tape.strain
        da = self.al*dl[0]+self.am*dm[0]
        db = self.bl*dl[0]+self.bm*dm[0]
        # Parameter dependence of the normal-velocity extension.
        ghost = np.zeros((self.exp.ny+4,self.exp.nx))
        for m in (1,2):
            ghost[2-m] = (2*m-1)/self.coeff*da*qxx[0]
        extra_raw = self.coeff*dy_extended(ghost,"yb")
        extra, extra_mem = pml(extra_raw,np.zeros_like(extra_raw),self.p["y"])
        out[PVYY] += extra_mem
        out[SXX] += self.lam*extra
        out[SYY] += (self.lam+2*self.mu)*extra
        out[SXX] += dl*(qxx+qyy)+2*dm*qxx
        out[SYY] += dl*(qxx+qyy)+2*dm*qyy
        out[SXY] += corner_jvp(self.mu,dm)*(qyx+qxy)
        # Top plane-stress law, including its full material derivative.
        out[SXX,0] -= self.lam[0]*extra[0]+dl[0]*(qxx[0]+qyy[0])+2*dm[0]*qxx[0]
        out[SXX,0] += db*qxx[0]
        out[SYY,0] = 0.
        return out, dtape.sample

    def reverse(self, statebar, tape, samplebar=None):
        """Hand-coded VJP, independently checked against basis-built matrices."""
        b = statebar.copy()
        b[SYY,0] = 0.
        sx,sy,ss = b[SXX].copy(),b[SYY].copy(),b[SXY].copy()
        bulk_sx = sx.copy(); bulk_sx[0] = 0.
        qxx,qyx,qxy,qyy = tape.strain
        gl = (bulk_sx+sy)*(qxx+qyy)
        gm = 2*(bulk_sx*qxx+sy*qyy)+corner_vjp(self.mu,ss*(qyx+qxy))
        gl[0] += self.bl*sx[0]*qxx[0]
        gm[0] += self.bm*sx[0]*qxx[0]
        xx = (self.lam+2*self.mu)*bulk_sx+self.lam*sy
        xx[0] += self.a*sx[0]
        yy = self.lam*bulk_sx+(self.lam+2*self.mu)*sy
        yx = self.mc*ss
        xy = self.mc*ss
        rawxy,b[PVXY] = pml_t(xy,b[PVXY],self.p["yh"])
        rawyy,b[PVYY] = pml_t(yy,b[PVYY],self.p["y"])
        vx,addxx,addyx,_ = self.yt(self.coeff*rawxy,"yf","vx",h_over_dt=1/self.coeff)
        vy,addxx2,_,abar = self.yt(self.coeff*rawyy,"yb","vy",alpha=self.alpha,qxx=qxx,h_over_dt=1/self.coeff)
        b[VX] += vx; b[VY] += vy
        xx += addxx+addxx2; yx += addyx
        gl[0] += self.al*abar; gm[0] += self.am*abar
        rawxx,b[PVXX] = pml_t(xx,b[PVXX],self.p["x"])
        rawyx,b[PVYX] = pml_t(yx,b[PVYX],self.p["xh"])
        b[VX] += self.coeff*self.xt(rawxx,"xb")
        b[VY] += self.coeff*self.xt(rawyx,"xf")
        if samplebar is not None:
            for r,(i,j) in enumerate(self.exp.receivers):
                b[VX,j-1,i-1] += samplebar[r,0]
                b[VY,j-1,i-1] += samplebar[r,1]
        raw = []
        for k,val,pname in zip((PSXX,PSXYY,PSXYX,PSYY),
                               (self.rx*b[VX],self.rx*b[VX],self.ry*b[VY],self.ry*b[VY]),
                               ("xh","y","x","yh")):
            corr,b[k] = pml_t(val,b[k],self.p[pname]); raw.append(corr)
        b[SXX] += self.coeff*self.xt(raw[0],"xf")
        b[SXY] += self.coeff*self.xt(raw[2],"xb")
        bx,*_ = self.yt(self.coeff*raw[1],"yb","sxy")
        by,*_ = self.yt(self.coeff*raw[3],"yf","syy")
        b[SXY] += bx; b[SYY] += by
        b[SYY,0] = 0.
        return b,gl,gm


def matrix_of(function, shape):
    """Actual forward matrix by unit-basis action; its T is the local oracle."""
    n = int(np.prod(shape))
    cols = []
    for i in range(n):
        e = np.zeros(n); e[i] = 1.
        cols.append(np.asarray(function(e.reshape(shape))).ravel())
    return np.stack(cols,axis=1)


def forward(exp, *, lam=None, mu=None, initial=None, start=0, stop=None, topology=(1,1)):
    op = SurfaceOperator(exp,lam=lam,mu=mu,topology=topology)
    z = op.zero() if initial is None else initial.copy()
    stop = exp.nt if stop is None else stop
    source = _source_samples(exp)
    i,j = exp.sources[0]; i-=1; j-=1
    tapes=[]; data=[]; checkpoints=[z.copy()]
    for k in range(start,stop):
        z,tape = op.step(z)
        # Same stress source timing and direct half-step receiver semantics as M9.
        z[SXX,j,i] += source[k]; z[SYY,j,i] += source[k]
        tapes.append(tape); data.append(tape.sample); checkpoints.append(z.copy())
    return np.asarray(data),tapes,checkpoints


def born(exp,tapes,dl,dm,*,topology=(1,1)):
    op = SurfaceOperator(exp,topology=topology); dz=op.zero(); data=[]
    for tape in tapes:
        dz,d = op.tangent(dz,tape,dl,dm); data.append(d)
    return np.asarray(data)


def adjoint(exp,tapes,data,*,topology=(1,1)):
    op = SurfaceOperator(exp,topology=topology); bar=op.zero()
    gl=np.zeros_like(exp.lam); gm=np.zeros_like(exp.mu)
    for tape,d in zip(reversed(tapes),reversed(data)):
        bar,l,m = op.reverse(bar,tape,d); gl+=l; gm+=m
    return gl,gm


def checkpoint_pack(op,state):
    """Existing M9d1/M9d2 content: five owned fields plus active memories."""
    parts=[state[:5].ravel()]
    for k,pname in enumerate(MEMORY_PROFILES,start=5):
        active=np.broadcast_to(op.p[pname].a,state[k].shape)!=0
        if np.any(state[k][~active]!=0):
            raise ValueError("omitted inactive checkpoint memories must be zero")
        parts.append(state[k][active])
    return np.concatenate(parts)


def checkpoint_restore(op,payload):
    z=op.zero();n=z[:5].size
    z[:5]=payload[:n].reshape(z[:5].shape)
    for k,pname in enumerate(MEMORY_PROFILES,start=5):
        active=np.broadcast_to(op.p[pname].a,z[k].shape)!=0
        count=int(np.count_nonzero(active))
        z[k][active]=payload[n:n+count];n+=count
    if n!=len(payload):
        raise ValueError("checkpoint payload layout mismatch")
    return z


@dataclass(frozen=True)
class CopyMap:
    """Staged overwrite graph. Explicit matrix and independent ADD/clear VJP."""
    size: int
    stages: tuple

    def forward(self,x):
        z=np.asarray(x).copy()
        for stage in self.stages:
            old=z.copy()
            for dest,src,sign in stage:
                z[dest]=0. if src is None else sign*old[src]
        return z

    def transpose(self,y):
        z=np.asarray(y).copy()
        for stage in reversed(self.stages):
            old=z.copy()
            for dest,_,_ in stage:
                z[dest]=0.
            for dest,src,sign in stage:
                if src is not None:
                    z[src]+=sign*old[dest]
        return z

    def matrix(self):
        b=np.eye(self.size)
        for stage in self.stages:
            old=b.copy()
            for dest,src,sign in stage:
                b[dest]=0. if src is None else sign*old[src]
        return b


def halo_surface_maps(nx,ny,px,py,kind):
    """All-field M9d2 staged x/y halo graph followed by top-owned traction map.

    Global outer halos wrap as in M9d2. Top syy/sxy ghosts override top wrap;
    x halos precede the boundary construction, including surface derivatives.
    """
    if nx%px or ny%py or nx//px<2 or ny//py<4:
        raise ValueError("uneven or shallow surface tile")
    lx,ly=nx//px,ny//py
    cells=(lx+4)*(ly+4); size=px*py*cells
    def idx(r,j,i): return r*cells+(j+2)*(lx+4)+i+2
    xs=[];ys=[];bs=[]
    for r in range(px*py):
        x,y=r%px,r//px
        for j in range(ly):
            for i in (-2,-1,lx,lx+1):
                gx=x*lx+i; xx=(gx%nx)//lx; ii=gx%lx
                xs.append((idx(r,j,i),idx(y*px+xx,j,ii),1.))
        for j in (-2,-1,ly,ly+1):
            gy=y*ly+j; yy=(gy%ny)//ly; jj=gy%ly
            for i in range(-2,lx+2):
                ys.append((idx(r,j,i),idx(yy*px+x,jj,i),1.))
        if y==0 and kind in ("syy","sxy"):
            for i in range(-2,lx+2):
                if kind=="syy": bs.append((idx(r,0,i),None,0.))
                for m in (1,2):
                    bs.append((idx(r,-m,i),idx(r,m if kind=="syy" else m-1,i),-1.))
    return CopyMap(size,(tuple(xs),tuple(ys))),CopyMap(size,(tuple(bs),))


def distributed_derivative(a,kind,topology,*,field=None,**kwargs):
    """Independent emulated ranks: Hx -> Hy -> top B -> owned FD4 -> gather.

    All external M9d2 wrap edges are retained except the top surface overwrite.
    Four top velocity rows are owned; surface x slopes use exchanged x halos.
    """
    ny,nx=a.shape;px,py=topology
    if topology==(1,1):
        if kind[0]=="x": return dx(a,kind)
        return dy_extended(extend(a,field,**kwargs),kind)
    lx,ly=nx//px,ny//py
    h,b=halo_surface_maps(nx,ny,px,py,field)
    local=np.zeros((px*py,ly+4,lx+4))
    for r in range(px*py):
        ox,oy=r%px*lx,r//px*ly
        local[r,2:ly+2,2:lx+2]=a[oy:oy+ly,ox:ox+lx]
    local=h.forward(local.ravel()).reshape(local.shape)
    if field in ("syy","sxy"):
        local=b.forward(local.ravel()).reshape(local.shape)
    elif field in ("vx","vy"):
        for r in range(px):
            ox=r*lx;col=slice(ox,ox+lx)
            args={}
            for key,val in kwargs.items():
                if isinstance(val,np.ndarray):
                    args[key]=val[:,col] if val.ndim==2 else val[col]
                else: args[key]=val
            ext=extend(a[:ly,col],field,**args)
            local[r,:2,2:lx+2]=ext[:2]
    out=np.zeros_like(a)
    for r in range(px*py):
        ox,oy=r%px*lx,r//px*ly
        if kind[0]=="x":
            values=sum(w*local[r,2:ly+2,2+o:2+o+lx] for o,w in TERMS[kind])
        else:
            values=sum(w*local[r,2+o:2+o+ly,2:lx+2] for o,w in TERMS[kind])
        out[oy:oy+ly,ox:ox+lx]=values
    return out


def distributed_derivative_t(a,kind,topology,*,field=None,**kwargs):
    """Owned FD4 scatter -> surface ADD/clear -> Hy.T -> Hx.T -> owners."""
    if topology==(1,1):
        if kind[0]=="x":
            return dx_t(a,kind),np.zeros_like(a),np.zeros_like(a),np.zeros(a.shape[1])
        return extend_t(dy_extended_t(a,kind),field,**kwargs)
    ny,nx=a.shape;px,py=topology;lx,ly=nx//px,ny//py
    h,b=halo_surface_maps(nx,ny,px,py,field)
    local=np.zeros((px*py,ly+4,lx+4))
    qx=np.zeros_like(a);qy=np.zeros_like(a);alphabar=np.zeros(nx)
    for r in range(px*py):
        ox,oy=r%px*lx,r//px*ly;owned=a[oy:oy+ly,ox:ox+lx]
        for o,w in TERMS[kind]:
            if kind[0]=="x": local[r,2:ly+2,2+o:2+o+lx]+=w*owned
            else: local[r,2+o:2+o+ly,2:lx+2]+=w*owned
    if field in ("syy","sxy"):
        local=b.transpose(local.ravel()).reshape(local.shape)
    elif field in ("vx","vy"):
        for r in range(px):
            ox=r*lx;col=slice(ox,ox+lx)
            args={}
            for key,val in kwargs.items():
                if isinstance(val,np.ndarray):
                    args[key]=val[:,col] if val.ndim==2 else val[col]
                else: args[key]=val
            ghost=np.zeros((ly+4,lx));ghost[:2]=local[r,:2,2:lx+2]
            physical,xbar,ybar,abar=extend_t(ghost,field,**args)
            local[r,:2,2:lx+2]=0.
            local[r,2:ly+2,2:lx+2]+=physical
            qx[:ly,col]+=xbar;qy[:ly,col]+=ybar;alphabar[col]+=abar
    local=h.transpose(local.ravel()).reshape(local.shape)
    out=np.zeros_like(a)
    for r in range(px*py):
        ox,oy=r%px*lx,r//px*ly
        out[oy:oy+ly,ox:ox+lx]=local[r,2:ly+2,2:lx+2]
    return out,qx,qy,alphabar


def boundary_metrics(stress):
    """Scale-aware traction and parity diagnostics of augmented stress arrays."""
    syy,sxy=stress
    scale=max(np.linalg.norm(syy),np.linalg.norm(sxy),np.finfo(float).tiny)
    parity=max(np.linalg.norm(syy[2-m]+syy[2+m]) for m in (1,2))
    parity=max(parity,*(np.linalg.norm(sxy[2-m]+sxy[2+m-1]) for m in (1,2)))
    shear=9/16*(sxy[1]+sxy[2])-1/16*(sxy[0]+sxy[3])
    regenerated=(extend(syy[2:-2],"syy"),extend(sxy[2:-2],"sxy"))
    closure=max(np.linalg.norm(syy-regenerated[0]),np.linalg.norm(sxy-regenerated[1]))
    return {"normal_traction":float(np.linalg.norm(syy[2])/scale),
            "shear_traction":float(np.linalg.norm(shear)/scale),
            "ghost_parity":float(parity/scale),
            "extension_closure":float(closure/scale)}
