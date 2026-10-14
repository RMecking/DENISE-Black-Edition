"""Only qualified FD8 Foundation and targeted canonical FD4 compatibility gates.
Expected exact map/rounding caches must already be sealed before these run.
"""
import ctypes as C
import gzip,json,hashlib,os,subprocess,tracemalloc,resource
from fractions import Fraction
from pathlib import Path
import numpy as np
import pytest
from tests.utilities import m9l1a_fd8_forward_reference as ref
from tests.physics import test_m9b1_elastic_psv_born_production as old

F=old.F32P
ROOT=Path(__file__).resolve().parents[2]
FLAGS=['-std=c99','-O2','-Wall','-Wextra','-Werror','-pedantic','-fPIC','-shared']

class Config(C.Structure):
    _fields_=[('struct_size',C.c_size_t),('abi_version',C.c_uint),('base',old.Config),
              ('coefficient_family',C.c_int),('field_boundary_policy',C.c_int),('material_boundary_policy',C.c_int)]

def pointer(a):return a.ctypes.data_as(F)
def array(bits,count):return np.full(count,bits,dtype=np.uint32).view(np.float32)

class Input:
    def __init__(self,nx=9,ny=9,nt=1,source=None,receivers=None):
        n=nx*ny;self.lam=array(ref.LAM_BITS,n);self.mu=array(ref.MU_BITS,n);self.rho=array(0x3fc00000,n)
        self.samples=np.zeros(nt,dtype=np.float32)
        source=source or (nx//2,ny//2);receivers=receivers or [source]
        self.i=np.asarray([x for x,y in receivers],dtype=np.int32)
        self.j=np.asarray([y for x,y in receivers],dtype=np.int32)
        b=old.Config();b.nx=nx;b.ny=ny;b.nt=nt;b.dh=1;b.dt=ref.float_value(ref.GAMMA_BITS)
        b.invmat1=3;b.fdorder=8;b.ndt=b.dtinv=b.mpi_size=1;b.receiver_components=2
        b.lambda_=pointer(self.lam);b.mu=pointer(self.mu);b.rho=pointer(self.rho)
        b.source_i,b.source_j=source;b.source_samples=pointer(self.samples)
        b.receiver_count=len(receivers);b.receiver_i=self.i.ctypes.data_as(old.I32P);b.receiver_j=self.j.ctypes.data_as(old.I32P)
        self.config=Config(C.sizeof(Config),1,b,1,1,1)
    def output(self,tail=0):return np.full(self.config.base.nt*self.config.base.receiver_count*2+tail,np.float32(-123.5),dtype=np.float32)

@pytest.fixture(scope='session')
def api(tmp_path_factory):
    cache=Path(os.environ['M9L1A_REFERENCE_FREEZE'])
    ledger=json.loads((cache/'reference-certificates.json').read_text())
    assert not ledger['candidate_outputs_read']
    assert ledger['source_reference_sha256']==hashlib.sha256(Path(ref.__file__).read_bytes()).hexdigest()
    for row in ledger['certificates']:
        assert hashlib.sha256((cache/(row['fixture']['id']+'-exact-and-witness.jsonl.gz')).read_bytes()).hexdigest()==row['expected_cache_sha256']
    freeze=json.loads(Path(os.environ['M9L1A_OPERATION_FREEZE']).read_text())
    assert freeze['flags']==FLAGS and freeze['reference_ledger_sha256']==hashlib.sha256((cache/'reference-certificates.json').read_bytes()).hexdigest()
    output=tmp_path_factory.mktemp('foundation')/'libfoundation.so'
    subprocess.run(['gcc',*FLAGS,'-I',str(ROOT/'include'),str(ROOT/'tests/utilities/m9l1a_fd8_forward_harness.c'),'-lm','-o',str(output)],check=True)
    a=C.CDLL(str(output))
    a.denise_elastic_psv_forward_fd8_create_v1.argtypes=[C.POINTER(Config),C.POINTER(C.c_void_p)]
    a.denise_elastic_psv_forward_fd8_create_v1.restype=C.c_int
    a.denise_elastic_psv_forward_fd8_nonlinear_v1.argtypes=[C.c_void_p,F,F,F,C.c_size_t]
    a.denise_elastic_psv_forward_fd8_nonlinear_v1.restype=C.c_int
    a.denise_elastic_psv_forward_fd8_destroy_v1.argtypes=[C.POINTER(C.c_void_p)]
    a.denise_elastic_psv_forward_fd8_last_error_v1.restype=C.c_char_p
    a.denise_fd8_test_capture.argtypes=[C.c_void_p,F,F,F,F]
    a.denise_fd8_test_capture.restype=C.c_int
    a.denise_fd8_test_derivative.argtypes=[C.c_void_p,F,C.c_int,F]
    a.denise_fd8_test_derivative.restype=C.c_int
    a.denise_fd8_test_maps.argtypes=[C.c_int,C.c_int,F,F,F]
    a.denise_fd8_test_maps.restype=C.c_int
    a.denise_fd8_test_fail_allocation.argtypes=[C.c_long]
    a.denise_fd8_test_set_round.argtypes=[C.c_int]
    a.denise_fd8_test_mxcsr.argtypes=[C.c_uint,C.c_int];a.denise_fd8_test_mxcsr.restype=C.c_uint
    a.denise_fd8_test_abi.argtypes=[C.c_int,C.c_int];a.denise_fd8_test_abi.restype=C.c_size_t
    a.denise_fd8_test_native_bytes.argtypes=[C.c_int];a.denise_fd8_test_native_bytes.restype=C.c_size_t
    a.denise_fd4_test_copy_maps.argtypes=[C.c_void_p,F]
    return a

def create(a,x):
    h=C.c_void_p();r=a.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),C.byref(h))
    assert r==0,a.denise_elastic_psv_forward_fd8_last_error_v1();return h
def close(a,h):a.denise_elastic_psv_forward_fd8_destroy_v1(C.byref(h));assert not h.value
def run(a,h,x,out=None,capacity=None,lam=None,mu=None):
    out=x.output() if out is None else out
    r=a.denise_elastic_psv_forward_fd8_nonlinear_v1(h,pointer(x.lam if lam is None else lam),
        pointer(x.mu if mu is None else mu),pointer(out),len(out) if capacity is None else capacity)
    return r,out

def exact_gate(actual,ev,predicted):
    bits=actual.view(np.uint32).tolist()
    error,norm=ref.compare(bits,ref.Vector(ev['n'],ev['e']))
    assert error<=Fraction(predicted), 'outside precomputed exact fixed-input native defect'
    if norm:assert error<=norm*Fraction(1,100000)**2
    else:assert error==0, 'fixture structurally zero vector'

def test_abi_version_prefix_and_ownership(api):
    assert api.denise_fd8_test_abi(0,0)==C.sizeof(old.Config)
    assert api.denise_fd8_test_abi(1,0)==C.sizeof(Config)
    for k,(name,_) in enumerate(Config._fields_):assert api.denise_fd8_test_abi(3,k)==getattr(Config,name).offset
    for k,(name,_) in enumerate(old.Config._fields_):assert api.denise_fd8_test_abi(2,k)==getattr(old.Config,name).offset
    assert api.denise_fd8_test_abi(4,0)==C.alignment(old.Config)
    assert api.denise_fd8_test_abi(5,0)==C.alignment(Config)
    # Only readable size prefix: rejection must not touch later fields.
    prefix=C.c_size_t(C.sizeof(C.c_size_t));h=C.c_void_p(999)
    assert api.denise_elastic_psv_forward_fd8_create_v1(C.cast(C.byref(prefix),C.POINTER(Config)),C.byref(h))==-1
    assert not h.value
    x=Input(nt=4);x.samples[0]=ref.float_value(ref.SOURCE_BITS);h=create(api,x)
    try:
        expected=run(api,h,x)[1];x.samples[:]=100;x.rho[:]=5;x.i[:]=0;x.j[:]=0
        np.testing.assert_array_equal(run(api,h,x)[1].view(np.uint32),expected.view(np.uint32))
    finally:close(api,h)
    api.denise_elastic_psv_forward_fd8_destroy_v1(None)

def test_all_finite_incidence_basis_and_transpose(api):
    x=Input(13,11);h=create(api,x);n=143;matrices,ce=ref.incidence(13,11)
    try:
        collected=[]
        for kind,rows in enumerate(matrices):
            matrix=np.zeros((n,n),dtype=np.float64)
            for column in range(n):
                v=np.zeros(n,np.float32);v[column]=1;out=np.zeros(n,np.float32)
                assert api.denise_fd8_test_derivative(h,pointer(v),kind,pointer(out))==0
                for p,row in enumerate(rows):
                    expected=sum(w for col,w in row if col==column)
                    ev=ref.Vector([expected],ce);err,norm=ref.compare([int(out.view(np.uint32)[p])],ev)
                    scale=Fraction(abs(expected),1<<-ce)
                    assert err<=(32*Fraction(1,1<<23)*scale)**2
                    matrix[p,column]=float(out[p])
            collected.append(matrix)
        np.testing.assert_array_equal(collected[0].T,-collected[1])
        np.testing.assert_array_equal(collected[2].T,-collected[3])
        # Expected-entry checks falsify outer-term omission and hidden wrap;
        # these are extra algebraic controls, not full Forward runs.
        assert collected[1][0,4]!=0
        assert collected[1][0,12]==0
        assert collected[2][0,3*13]==ref.float_value(ref.COEFFICIENT_BITS[3])
        xx=[(-1)**p for p in range(n)];yy=[(p%5)-2 for p in range(n)]
        for rows,matrix in zip(matrices,collected):
            # Independent exact dyadic expected dot, rather than agreement of
            # a common wrong forward/transpose pair. These small integer inputs
            # retain exact binary64 product/sum representability.
            expected=sum(Fraction(w,1<<-ce)*xx[col]*yy[row] for row,entries in enumerate(rows) for col,w in entries)
            dx=matrix@np.asarray(xx,np.float64);dty=matrix.T@np.asarray(yy,np.float64)
            left=Fraction.from_float(float(np.dot(dx,yy)));right=Fraction.from_float(float(np.dot(xx,dty)))
            assert left==expected and right==expected
    finally:close(api,h)

def test_replicated_material_helper_including_corners(api):
    nx,ny=13,11;n=nx*ny
    tag=np.asarray([1+(i+2*j)/32 for j in range(ny) for i in range(nx)],dtype=np.float32)
    out=np.empty(3*n,np.float32)
    assert api.denise_fd8_test_maps(nx,ny,pointer(tag),pointer(tag),pointer(out))==0
    bits=tag.view(np.uint32)
    for j in range(ny):
        for i in range(nx):
            p=j*nx+i;ip=min(i+1,nx-1);jp=min(j+1,ny-1)
            corner=[int(bits[a*nx+b]) for a,b in ((j,i),(j,ip),(jp,i),(jp,ip))]
            mx,h=ref.material_maps(corner,[int(bits[p]),int(bits[j*nx+ip])])
            my,_=ref.material_maps(corner,[int(bits[p]),int(bits[jp*nx+i])])
            assert out.view(np.uint32)[p]==mx and out.view(np.uint32)[n+p]==my and out.view(np.uint32)[2*n+p]==h

@pytest.mark.parametrize('name',('B1','B3'))
def test_exact_finite_substeps_source_and_rounding_witness(api,name):
    cache=Path(os.environ['M9L1A_REFERENCE_FREEZE'])
    with gzip.open(cache/(name+'-exact-and-witness.jsonl.gz'),'rt') as f:
        metadata=json.loads(next(f));case=metadata['fixture']
        x=Input(case['nx'],case['ny'],case['nt'],case['source'],case['receivers'])
        if name=='B3':x.samples[0]=ref.float_value(ref.SOURCE_BITS)
        h=create(api,x);n=case['nx']*case['ny'];nt=case['nt']
        initial=np.asarray(metadata['initial_bits'],np.uint32).view(np.float32)
        fields=np.zeros((nt,4,5,n),np.float32);q=np.zeros((nt,4,n),np.float32);out=x.output()
        try:
            assert api.denise_fd8_test_capture(h,pointer(initial),pointer(out),pointer(fields),pointer(q))==0
            for line in f:
                row=json.loads(line);t=row['time']
                for v in row['states']:
                    actual=fields[t,v['stage'],v['component']]
                    np.testing.assert_array_equal(actual.view(np.uint32),v['witness_bits'])
                    exact_gate(actual,v['exact'],v['defect_norm2'])
                for c in range(4):
                    np.testing.assert_array_equal(q[t,c].view(np.uint32),row['strain_witness_bits'][c])
                    predicted=ref.compare(row['strain_witness_bits'][c],ref.Vector(row['strain_exact'][c]['n'],row['strain_exact'][c]['e']))[0]
                    exact_gate(q[t,c],row['strain_exact'][c],str(predicted))
                actual=out[t*len(case['receivers'])*2:(t+1)*len(case['receivers'])*2]
                np.testing.assert_array_equal(actual.view(np.uint32),row['data_witness_bits'])
                exact=ref.vectors_from_bits([row['data_witness_bits']])[0]
                # Data target assembled directly from independently exact momentum.
                e=min(v['e'] for v in row['data_exact'])
                target=ref.Vector([v['n']<<(v['e']-e) for v in row['data_exact']],e)
                expected_error,_=ref.compare(row['data_witness_bits'],target)
                exact_gate(actual,target.payload(),str(expected_error))
            if name=='B3':
                assert np.all(out[:16].view(np.uint32)==0)
                sample=out[16:32].reshape(8,2)
                assert sample[0,0]<0 and sample[1,0]>0 and sample[2,1]<0 and sample[3,1]>0
                assert np.all(sample[4:]==0)
                for _ in range(2):np.testing.assert_array_equal(run(api,h,x)[1].view(np.uint32),out.view(np.uint32))
                fresh=create(api,x)
                try:np.testing.assert_array_equal(run(api,fresh,x)[1].view(np.uint32),out.view(np.uint32))
                finally:close(api,fresh)
        finally:close(api,h)

@pytest.mark.parametrize('field,value',[
    ('nx',8),('ny',8),('nt',0),('dt',0),('dt',-1),('dh',0),('dh',float('nan')),('dt',float('inf')),
    ('dt',1/3),('free_surface',1),('cpml_enabled',1),('cpml_enabled',-1),('mpi_size',2),('fdorder',4),
    ('source_i',9),('receiver_components',1),('boundary',1),('fw',1),('l',1),('invmat1',1),('ndt',2),('dtinv',2)])
def test_invalid_config_before_publication(api,field,value):
    x=Input();setattr(x.config.base,field,value);h=C.c_void_p(123)
    assert api.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),C.byref(h))==-1
    assert not h.value

@pytest.mark.parametrize('field,value',[
    ('struct_size',0),('struct_size',C.sizeof(Config)+1),('abi_version',0),('abi_version',2),
    ('coefficient_family',0),('field_boundary_policy',0),('material_boundary_policy',0)])
def test_invalid_api_identities(api,field,value):
    x=Input();setattr(x.config,field,value);h=C.c_void_p()
    assert api.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),C.byref(h))==-1
    assert not h.value

def test_valid_negative_lambda_subnormals_and_invalid_material(api):
    for lam,mu,rho in ((-1.6875,3.375,1.5),(0,ref.float_value(1),1.5)):
        x=Input();x.lam[:]=lam;x.mu[:]=mu;x.rho[:]=rho;h=create(api,x)
        try:r,y=run(api,h,x);assert r==0 and np.all(y==0)
        finally:close(api,h)
    for field,value in (('mu',0),('mu',-1),('rho',0),('rho',-1),('lam',-3.375),('lam',float('nan')),
                        ('mu',float('inf')),('rho',ref.float_value(1)),('mu',ref.float_value(0x7f000000))):
        x=Input();getattr(x,field)[:]=value;h=C.c_void_p()
        assert api.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),C.byref(h))==-1
        assert not h.value
    x=Input();x.mu[3]*=2;h=C.c_void_p()
    assert api.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),C.byref(h))==-1
    x=Input();x.samples[0]=np.nan
    assert api.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),C.byref(h))==-1

def test_atomic_capacity_trial_failures_and_valid_retry(api):
    x=Input();h=create(api,x)
    try:
        out=x.output(1);before=out.copy();r,_=run(api,h,x,out,capacity=1)
        assert r==-1 and out.tobytes()==before.tobytes()
        for lam,mu in ((array(0x7fc00000,81),x.mu),(x.lam,np.zeros(81,np.float32)),
                       (np.full(81,-3.375,np.float32),x.mu),(np.full(81,10000,np.float32),x.mu)):
            r,_=run(api,h,x,out,lam=lam,mu=mu);assert r==-1 and out.tobytes()==before.tobytes()
        r,_=run(api,h,x,out);assert r==0 and np.all(out[:2]==0) and out[-1]==before[-1]
        for t in range(19):
            out=x.output();before=out.copy();api.denise_fd8_test_fail_allocation(t)
            r,_=run(api,h,x,out);api.denise_fd8_test_fail_allocation(-1)
            assert r==-1 and out.tobytes()==before.tobytes()
            assert run(api,h,x)[0]==0
        assert api.denise_elastic_psv_forward_fd8_nonlinear_v1(h,None,pointer(x.mu),pointer(out),2)==-1
        assert api.denise_elastic_psv_forward_fd8_nonlinear_v1(None,pointer(x.lam),pointer(x.mu),pointer(out),2)==-1
    finally:api.denise_fd8_test_fail_allocation(-1);close(api,h)

def test_constructor_allocation_overflow_cleanup(api):
    x=Input()
    for t in range(23):
        h=C.c_void_p(999);api.denise_fd8_test_fail_allocation(t)
        r=api.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),C.byref(h))
        api.denise_fd8_test_fail_allocation(-1)
        assert r==-1 and not h.value
        h=create(api,x);close(api,h)
    x.config.base.nx=x.config.base.ny=2**31-1;h=C.c_void_p()
    assert api.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),C.byref(h))==-1
    assert not h.value

def test_environment_failure_is_atomic_and_does_not_repair_process(api):
    libm=C.CDLL('libm.so.6');libm.fegetround.argtypes=[];libm.fegetround.restype=C.c_int
    saved=api.denise_fd8_test_mxcsr(0,0)
    assert libm.fegetround()==0  # FE_TONEAREST on this qualified x86_64 environment.
    x=Input();h=create(api,x);records=[]
    try:
        for mode in (1,2,3):
            y=x.output();before=y.copy()
            try:
                assert api.denise_fd8_test_set_round(mode)==0
                assert run(api,h,x,y)[0]==-1
                c=C.c_void_p();assert api.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),C.byref(c))==-1
            finally:
                assert api.denise_fd8_test_set_round(0)==0
                api.denise_fd8_test_mxcsr(saved,1)
            assert y.tobytes()==before.tobytes()
        for flag in (0x40,0x8000):
            y=x.output();before=y.copy()
            try:
                api.denise_fd8_test_mxcsr(saved|flag,1);assert run(api,h,x,y)[0]==-1
                assert api.denise_fd8_test_mxcsr(0,0)&flag
            finally:api.denise_fd8_test_mxcsr(saved,1)
            assert y.tobytes()==before.tobytes()
        for rc in (0x2000,0x4000,0x6000):
            y=x.output();before=y.tobytes();c=C.c_void_p(999)
            rejected=(saved&~0x6000)|rc
            try:
                api.denise_fd8_test_mxcsr(rejected,1)
                x87_before=libm.fegetround();assert x87_before==0
                create_status=api.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),C.byref(c))
                assert create_status==-1 and not c.value
                after_create=api.denise_fd8_test_mxcsr(0,0);assert after_create==rejected
                nonlinear_status=run(api,h,x,y)[0];assert nonlinear_status==-1
                after_nonlinear=api.denise_fd8_test_mxcsr(0,0);assert after_nonlinear==rejected
                assert libm.fegetround()==0 and y.tobytes()==before
                records.append({'SSE_RC':rc,'x87_round_before':x87_before,'x87_round_after':libm.fegetround(),
                    'requested_MXCSR':rejected,'after_create_MXCSR':after_create,'after_nonlinear_MXCSR':after_nonlinear,
                    'create_status':create_status,'handle_is_null':not c.value,'nonlinear_status':nonlinear_status,
                    'output_before_SHA256':hashlib.sha256(before).hexdigest(),
                    'output_after_SHA256':hashlib.sha256(y.tobytes()).hexdigest()})
            finally:
                api.denise_fd8_test_mxcsr(saved,1)
            assert api.denise_fd8_test_mxcsr(0,0)==saved and libm.fegetround()==0
        assert run(api,h,x)[0]==0
        fresh=create(api,x)
        try:assert run(api,fresh,x)[0]==0
        finally:close(api,fresh)
        if os.environ.get('M9L1A_H1_EVIDENCE'):
            path=Path(os.environ['M9L1A_H1_EVIDENCE'])/'environment-regression.json'
            with path.open('x',encoding='utf-8') as stream:
                json.dump({'saved_MXCSR':saved,'x87_RNE_confirmed':True,'SSE_non_RNE':records,
                           'retained_x87_directed_rounding_and_FTZ_DAZ_tests':'PASS',
                           'RNE_restoration_creation_and_Forward':'PASS'},stream,indent=2)
    finally:
        try:
            assert api.denise_fd8_test_set_round(0)==0
        finally:
            api.denise_fd8_test_mxcsr(saved,1)
            close(api,h)

def test_numerical_failure_including_final_source_publishes_nothing(api):
    x=Input(nt=2);x.samples[:]=[ref.float_value(0x7e800000),ref.float_value(0x7f7fffff)];h=create(api,x)
    try:
        y=x.output();before=y.copy();assert run(api,h,x,y)[0]==-1
        assert y.tobytes()==before.tobytes()
    finally:close(api,h)
    x=Input();x.samples[0]=ref.float_value(0x7f7fffff);h=create(api,x)
    try:assert run(api,h,x)[0]==0  # No hypothetical future timestep.
    finally:close(api,h)
    x=Input(nt=2);h=create(api,x);n=81
    try:
        initial=np.zeros((5,n),np.float32);initial[2,40]=ref.float_value(0x7f7fffff)
        y=x.output();before=y.copy()
        assert api.denise_fd8_test_capture(h,pointer(initial),pointer(y),None,None)==-1
        assert y.tobytes()==before.tobytes() and run(api,h,x)[0]==0
        initial[2,40]=np.nan
        assert api.denise_fd8_test_capture(h,pointer(initial),pointer(y),None,None)==-1
    finally:close(api,h)

@pytest.mark.parametrize('name',('B1','B3'))
def test_foundation_simultaneous_working_memory_accounting(api,name):
    """Same approved cases, instrumented to close required resource evidence."""
    cache=Path(os.environ['M9L1A_REFERENCE_FREEZE']);baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    api.denise_fd8_test_reset_native_peak();tracemalloc.start()
    with gzip.open(cache/(name+'-exact-and-witness.jsonl.gz'),'rt') as f:
        metadata=json.loads(next(f));case=metadata['fixture'];n=case['nx']*case['ny'];nt=case['nt']
        x=Input(case['nx'],case['ny'],nt,case['source'],case['receivers'])
        if name=='B3':x.samples[0]=ref.float_value(ref.SOURCE_BITS)
        h=create(api,x);initial=np.asarray(metadata['initial_bits'],np.uint32).view(np.float32)
        fields=np.zeros((nt,4,5,n),np.float32);q=np.zeros((nt,4,n),np.float32);out=x.output()
        try:
            assert api.denise_fd8_test_capture(h,pointer(initial),pointer(out),pointer(fields),pointer(q))==0
            for line in f:
                row=json.loads(line);t=row['time']
                for v in row['states']:exact_gate(fields[t,v['stage'],v['component']],v['exact'],v['defect_norm2'])
            python_peak=tracemalloc.get_traced_memory()[1]
            native_peak=api.denise_fd8_test_native_bytes(1);allocations=api.denise_fd8_test_native_bytes(2)
        finally:close(api,h);tracemalloc.stop()
    assert api.denise_fd8_test_native_bytes(0)==0
    # Native allocator headers/alignment allowance is an engineering estimate,
    # distinct from measured requested bytes and observed full-process RSS.
    estimated_overhead=64*allocations
    total=python_peak+native_peak+estimated_overhead
    assert total<16*1024*1024,'case-specific working allocation target overrun'
    destination=Path(os.environ['M9L1A_EVIDENCE'])/(name+'-working-memory.json')
    destination.write_text(json.dumps({'case':name,'measured_tracemalloc_peak_bytes':python_peak,
        'measured_native_requested_peak_bytes':native_peak,'native_live_allocation_peak':allocations,
        'estimated_native_allocator_header_alignment_bytes':estimated_overhead,'conservative_sum_peak_bytes':total,
        'full_process_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'before_case_peak_RSS_KiB':baseline,
        'simultaneous_native_reference_snapshot_and_temporary_buffers_accounted':True,
        'serialized_payload_not_used_as_working_peak':True},indent=2)+'\n')

@pytest.fixture(scope='session')
def fd4_pair(tmp_path_factory):
    baseline=Path(os.environ['M9L1A_BASELINE_SOURCE'])
    assert hashlib.sha256((baseline/'src/PSV/elastic_psv_born.c').read_bytes()).hexdigest()=='8ba0d4239fdbe4a3cd3f22d6c260a25f1159ee621c594261792e5aded6b94d5c'
    b=old.born_library.__wrapped__(tmp_path_factory,baseline)
    a=old.born_library.__wrapped__(tmp_path_factory,ROOT)
    for library in (b,a):
        library.denise_elastic_psv_born_set_replay_segments.argtypes=[C.c_void_p,C.c_int]
        library.denise_elastic_psv_born_set_replay_segments.restype=C.c_int
        library.denise_elastic_psv_born_checkpoint_roundtrip.argtypes=[C.c_void_p,C.c_int]
        library.denise_elastic_psv_born_checkpoint_roundtrip.restype=C.c_int
    return b,a

@pytest.mark.parametrize('case',('interior','active_cpml'))
def test_fd4_targeted_baseline_candidate_bytes_and_canonical_oracle(api,fd4_pair,case):
    baseline,candidate=fd4_pair;exp=old.make_experiment(case)
    b=old.Operator(baseline,exp);c=old.Operator(candidate,exp)
    def identical(left,right):assert left.tobytes()==right.tobytes()
    try:
        yb=b.prepare();yc=c.prepare();identical(yb,yc)
        oracle=old.nonlinear_forward(exp,0,save_strain=True)
        old._assert_production_close(yc,oracle.data,2e-6)
        mb=np.empty((6,exp.ny,exp.nx),np.float32);mc=np.empty_like(mb)
        api.denise_fd4_test_copy_maps(b.context,pointer(mb));api.denise_fd4_test_copy_maps(c.context,pointer(mc));identical(mb,mc)
        for time in range(exp.nt):
            sb=np.empty((4,exp.ny,exp.nx),np.float32);sc=np.empty_like(sb)
            assert baseline.denise_elastic_psv_born_copy_strain(b.context,time,pointer(sb))==0
            assert candidate.denise_elastic_psv_born_copy_strain(c.context,time,pointer(sc))==0
            identical(sb,sc)
        dl,dm=old.deterministic_model_direction(exp);zero=np.zeros_like(dl)
        for l,m in ((dl,zero),(zero,dm),(dl,dm)):
            jb=b.j(l,m);jc=c.j(l,m);identical(jb,jc)
            old._assert_production_close(jc,old.born_forward(exp,oracle,l,m),1e-5)
        for component in (0,1,None):
            data=old.deterministic_data(exp,component=component);gb=b.jt(data);gc=c.jt(data)
            identical(gb[0],gc[0]);identical(gb[1],gc[1])
            expected=old.born_adjoint(exp,oracle,data)
            old._assert_production_close(gc[0],expected.image_lambda_raw,6e-5)
            old._assert_production_close(gc[1],expected.image_mu_raw,6e-5)
        nb=np.empty_like(yb);nc=np.empty_like(yc)
        assert baseline.denise_elastic_psv_born_nonlinear(b.context,pointer(b.lam),pointer(b.mu),pointer(nb))==0
        assert candidate.denise_elastic_psv_born_nonlinear(c.context,pointer(c.lam),pointer(c.mu),pointer(nc))==0
        identical(nb,nc)
        if exp.cpml:assert candidate.denise_elastic_psv_born_cpml_memory_peak(c.context)>0
    finally:b.close();c.close()

@pytest.mark.parametrize('case,segments',(('interior',7),('active_cpml',32)))
def test_fd4_named_full_segmented_and_checkpoint_representatives(fd4_pair,case,segments):
    baseline,candidate=fd4_pair;exp=old.make_experiment(case)
    operators=[old.Operator(lib,exp) for lib in (baseline,candidate,baseline,candidate)]
    try:
        for lib,operator in zip((baseline,candidate),operators[2:]):
            assert lib.denise_elastic_psv_born_set_replay_segments(operator.context,segments)==0
        data=[o.prepare() for o in operators]
        assert all(a.tobytes()==data[0].tobytes() for a in data[1:])
        # Preserve the exact canonical representative's endpoint/boundary probes;
        # each SEG copy replays a segment, so do not expand it to all NT slots.
        probe={0,exp.nt-1}
        for s in range(segments):
            start,end=s*exp.nt//segments,(s+1)*exp.nt//segments
            probe.update((start,end-1))
            if end-start>1:probe.add(end-2)
            if start>0:probe.add(start-1)
        for time in sorted(probe):
            strains=[]
            for library,operator in zip((baseline,candidate,baseline,candidate),operators):
                s=np.empty((4,exp.ny,exp.nx),np.float32)
                assert library.denise_elastic_psv_born_copy_strain(operator.context,time,pointer(s))==0;strains.append(s)
            assert all(s.tobytes()==strains[0].tobytes() for s in strains[1:])
        dl,dm=old.deterministic_model_direction(exp);zero=np.zeros_like(dl)
        for l,m in ((dl,zero),(zero,dm),(dl,dm)):
            products=[o.j(l,m) for o in operators]
            assert all(v.tobytes()==products[0].tobytes() for v in products[1:])
        for component in (0,1,None):
            residual=old.deterministic_data(exp,component=component);images=[o.jt(residual) for o in operators]
            for channel in (0,1):assert all(v[channel].tobytes()==images[0][channel].tobytes() for v in images[1:])
        if case=='active_cpml':
            for library,operator in zip((baseline,candidate),operators[:2]):
                assert library.denise_elastic_psv_born_checkpoint_roundtrip(operator.context,347)==0
    finally:
        for o in operators:o.close()

def test_a1_rational_moments_and_stored_bits():
    # The already accepted Taylor identities are tested, not refitted/reopened.
    weights=(Fraction(1225,1024),Fraction(-245,3072),Fraction(49,5120),Fraction(-5,7168))
    expected={1:Fraction(1),3:Fraction(0),5:Fraction(0),7:Fraction(0),9:Fraction(-11025,256)}
    for power,value in expected.items():
        assert sum(2*c*(Fraction(m)-Fraction(1,2))**power for m,c in enumerate(weights,1))==value
    assert [ref.round_ratio(c.numerator,c.denominator) for c in weights]==list(ref.COEFFICIENT_BITS)

def test_b4_required_pointers_receiver_and_count_checks(api):
    # These are the qualified required-pointer/overflow/owned-receiver categories,
    # not additional wave propagation fixtures.
    x=Input();h=C.c_void_p(99)
    assert api.denise_elastic_psv_forward_fd8_create_v1(None,C.byref(h))==-1 and not h.value
    assert api.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),None)==-1
    for name in ('lambda_','mu','rho','source_samples','receiver_i','receiver_j'):
        x=Input();setattr(x.config.base,name,None);h=C.c_void_p()
        assert api.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),C.byref(h))==-1 and not h.value
    for coords,value in (('i',9),('i',-1),('j',9),('j',-1)):
        x=Input();getattr(x,coords)[0]=value;h=C.c_void_p()
        assert api.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),C.byref(h))==-1 and not h.value
    for dt,dh in ((ref.float_value(1),2),(ref.float_value(0x7f7fffff),ref.float_value(1))):
        x=Input();x.config.base.dt=dt;x.config.base.dh=dh;h=C.c_void_p()
        assert api.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),C.byref(h))==-1 and not h.value
    x=Input();x.config.base.nt=x.config.base.receiver_count=2**31-1;h=C.c_void_p()
    assert api.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),C.byref(h))==-1 and not h.value
    x=Input();h=create(api,x);y=x.output();before=y.copy()
    try:
        assert api.denise_elastic_psv_forward_fd8_nonlinear_v1(h,pointer(x.lam),None,pointer(y),2)==-1
        assert y.tobytes()==before.tobytes()
        assert api.denise_elastic_psv_forward_fd8_nonlinear_v1(h,pointer(x.lam),pointer(x.mu),None,2)==-1
    finally:close(api,h)

def test_b4_independent_stored_map_cfl_neighbors_without_propagation(api):
    one64=0x3ff0000000000000;two64=0x4000000000000000
    guard=0x3fe47ae147ae147a
    def upper(gamma):
        # Independent exact IEEE witness for the qualified RN64+successor DAG.
        s=0
        for c in ref.COEFFICIENT_BITS:s=ref.op('add',s,ref.widen(c&0x7fffffff),64)
        s=ref.op('mul',two64,s,64)
        r=ref.op('add',ref.widen(ref.LAM_BITS),ref.op('mul',two64,ref.widen(ref.MU_BITS),64),64)+1
        g2=ref.op('mul',ref.widen(gamma),ref.widen(gamma),64)+1
        s2=ref.op('mul',s,s,64)+1
        t1=ref.op('mul',g2,ref.widen(ref.MAP_BITS),64)+1
        t2=ref.op('mul',t1,r,64)+1
        t3=ref.op('mul',t2,s2,64)+1
        return ref.op('div',t3,two64,64)+1
    assert ref.fraction(upper(ref.GAMMA_BITS),64)==Fraction(2682583629393415,144115188075855872)
    # Expected metadata is derived without any candidate call/output; only then
    # exercise the adjacent representable input constructors. No Forward runs.
    low,high=ref.GAMMA_BITS,0x3eaaaaab
    assert upper(low)<=guard<upper(high)
    while high-low>1:
        mid=(low+high)//2
        if upper(mid)<=guard:low=mid
        else:high=mid
    predictions=[(low,True,upper(low)),(high,False,upper(high))]
    record={'criterion':'Q_U <= greatest binary64 <=16/25','computed_guard_bits':hex(guard),
            'expected_inputs_frozen_before_candidate_constructor_calls':True,
            'inputs':[{'gamma_bits':hex(g),'Q_upper_bits':hex(q),'admit':ok} for g,ok,q in predictions]}
    path=Path(os.environ['M9L1A_EVIDENCE'])/'cfl-boundary-inputs.json'
    with path.open('x') as f:json.dump(record,f,indent=2)
    for gamma,admit,q in predictions:
        x=Input();x.config.base.dt=ref.float_value(gamma);h=C.c_void_p()
        result=api.denise_elastic_psv_forward_fd8_create_v1(C.byref(x.config),C.byref(h))
        assert result==(0 if admit else -1)
        if admit:close(api,h)
        else:assert not h.value

def test_normal_build_b3_public_data_matches_frozen_rounding_witness():
    """The ordinary library has no private initialization/snapshot hooks."""
    a=C.CDLL(os.environ['M9L1A_NORMAL_LIBRARY'])
    a.denise_elastic_psv_forward_fd8_create_v1.argtypes=[C.POINTER(Config),C.POINTER(C.c_void_p)]
    a.denise_elastic_psv_forward_fd8_create_v1.restype=C.c_int
    a.denise_elastic_psv_forward_fd8_nonlinear_v1.argtypes=[C.c_void_p,F,F,F,C.c_size_t]
    a.denise_elastic_psv_forward_fd8_nonlinear_v1.restype=C.c_int
    a.denise_elastic_psv_forward_fd8_destroy_v1.argtypes=[C.POINTER(C.c_void_p)]
    a.denise_elastic_psv_forward_fd8_last_error_v1.restype=C.c_char_p
    cache=Path(os.environ['M9L1A_REFERENCE_FREEZE'])
    with gzip.open(cache/'B3-exact-and-witness.jsonl.gz','rt') as f:
        header=json.loads(next(f));case=header['fixture']
        x=Input(48,48,4,case['source'],case['receivers']);x.samples[0]=ref.float_value(ref.SOURCE_BITS)
        expected=np.asarray([b for line in f for b in json.loads(line)['data_witness_bits']],np.uint32)
    h=create(a,x)
    try:
        result,data=run(a,h,x);assert result==0
        np.testing.assert_array_equal(data.view(np.uint32),expected)
    finally:close(a,h)
