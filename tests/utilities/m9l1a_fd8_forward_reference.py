"""Independent sparse finite incidence/dyadic reference and scalar rounding DAG.

No candidate imports, kernels, or outputs define this reference. Primary physics
is exact sparse A_G/B_G composition. The separately rounded witness is not its
sole oracle. IEEE nodes use integer quotient/remainder and ties-to-even.
"""
from __future__ import annotations
import gzip,json,struct,hashlib,sys,time,tracemalloc,resource
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

COEFFICIENT_BITS=(0x3f992000,0xbda35555,0x3c1ccccd,0xba36db6e)
MAP_BITS=0x3f2aaaab
GAMMA_BITS=0x3ccccccd
LAM_BITS=0x40d80000
MU_BITS=0x40580000
SOURCE_BITS=0x3a83126f
COUNTS={}

def parts(bits,width=32):
    precision,ebits=(24,8) if width==32 else (53,11)
    frac=precision-1;bias=(1<<(ebits-1))-1
    sign=bits>>(width-1);exponent=(bits>>frac)&((1<<ebits)-1)
    mantissa=bits&((1<<frac)-1)
    if exponent==(1<<ebits)-1:raise ArithmeticError('nonfinite reference node')
    value=mantissa if exponent==0 else (1<<frac)|mantissa
    return (-value if sign else value),(1-bias-frac if exponent==0 else exponent-bias-frac)

def fraction(bits,width=32):
    n,e=parts(bits,width)
    return Fraction(n<<e,1) if e>=0 else Fraction(n,1<<-e)

def round_ratio(n,den=1,width=32,negative_zero=False):
    """Exact rational to IEEE RNE including subnormals, overflow, and signed zero."""
    precision,ebits=(24,8) if width==32 else (53,11)
    bias=(1<<(ebits-1))-1;frac=precision-1
    negative=(n<0)^(den<0);n=abs(n);den=abs(den)
    if not den:raise ZeroDivisionError
    if not n:return (1<<(width-1)) if negative_zero else 0
    exponent=n.bit_length()-den.bit_length()
    if (n < den<<exponent) if exponent>=0 else (n<<-exponent < den):exponent-=1
    unit=max(exponent-frac,1-bias-frac)
    a,b=(n,den<<unit) if unit>=0 else (n<<-unit,den)
    q,r=divmod(a,b)
    if 2*r>b or (2*r==b and q&1):q+=1
    sign=int(negative)<<(width-1)
    if q==0:return sign
    if q==(1<<precision):q>>=1;exponent+=1;unit+=1
    if exponent>bias:return sign|(((1<<ebits)-1)<<frac)
    if unit==1-bias-frac and q<(1<<frac):return sign|q
    return sign|((unit+frac+bias)<<frac)|(q-(1<<frac))

def encode_dyadic(n,e,width=32,negative_zero=False):
    return round_ratio(n<<e,1,width,negative_zero) if e>=0 else round_ratio(n,1<<-e,width,negative_zero)

def op(kind,a,b,width=32):
    key=f'RN{width}_{kind}';COUNTS[key]=COUNTS.get(key,0)+1
    an,ae=parts(a,width);bn,be=parts(b,width);signbit=1<<(width-1)
    if kind in ('add','sub'):
        if kind=='sub':b^=signbit;bn=-bn
        e=min(ae,be);n=(an<<(ae-e))+(bn<<(be-e))
        return encode_dyadic(n,e,width,not n and bool(a&b&signbit))
    if kind=='mul':
        return encode_dyadic(an*bn,ae+be,width,not (an*bn) and bool((a^b)&signbit))
    if kind=='div':
        e=ae-be;n,d=an,bn
        if e>=0:n<<=e
        else:d<<=-e
        return round_ratio(n,d,width,not n and bool((a^b)&signbit))
    raise ValueError(kind)

def widen(bits):return encode_dyadic(*parts(bits),64, bool(bits==0x80000000))
def narrow(bits):return encode_dyadic(*parts(bits,64),32, bool(bits==0x8000000000000000))
def input_bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def float_value(bits):return struct.unpack('<f',struct.pack('<I',bits))[0]

def material_maps(mu,rho):
    """Maps are independently rounded from exact dyadic input bits."""
    one=0x3ff0000000000000;two=0x4000000000000000;four=0x4010000000000000
    face=narrow(op('div',two,op('add',widen(rho[0]),widen(rho[1]),64),64))
    reciprocals=[op('div',one,widen(m),64) for m in mu]
    z=reciprocals[0]
    for r in reciprocals[1:]:z=op('add',z,r,64)
    corner=narrow(op('div',four,z,64))
    return face,corner

@dataclass
class Vector:
    numerator:list[int]
    exponent:int
    def plus(self,other):
        e=min(self.exponent,other.exponent);a=self.exponent-e;b=other.exponent-e
        return Vector([(x<<a)+(y<<b) for x,y in zip(self.numerator,other.numerator)],e)
    def scale(self,bits):
        n,e=parts(bits);return Vector([n*x for x in self.numerator],self.exponent+e)
    def payload(self):return {'n':self.numerator,'e':self.exponent}

def vectors_from_bits(fields):
    result=[]
    for field in fields:
        decoded=[parts(b) for b in field];e=min(k[1] for k in decoded)
        result.append(Vector([n<<(a-e) for n,a in decoded],e))
    return result

def incidence(nx,ny):
    """Rows constructed from stated signed offsets, not production basis output.
    Order keys: x-back,x-forward,y-back,y-forward. Material extension is separate.
    """
    coefficients=[parts(b) for b in COEFFICIENT_BITS]
    ce=min(e for _,e in coefficients)
    weights=[n<<(e-ce) for n,e in coefficients]
    signed_weights=[(w,-w) for w in weights]
    columns=list(range(nx*ny))
    matrices=[]
    for axis,forward in ((0,False),(0,True),(1,False),(1,True)):
        rows=[]
        for j in range(ny):
            for i in range(nx):
                row=[]
                for m,pair in enumerate(signed_weights,1):
                    offsets=(m,1-m) if forward else (m-1,-m)
                    for offset,weight in zip(offsets,pair):
                        x=i+offset if axis==0 else i;y=j+offset if axis==1 else j
                        if 0<=x<nx and 0<=y<ny:row.append((columns[y*nx+x],weight))
                rows.append(tuple(row))
        matrices.append(rows)
    return matrices,ce

def apply_matrix(matrix,coefficient_exponent,vector):
    return Vector([sum(w*vector.numerator[col] for col,w in row) for row in matrix],coefficient_exponent+vector.exponent)

def exact_step(fields,matrices,ce,source_index,source):
    """Independent linear momentum/constitutive block-map composition."""
    vx,vy,xx,yy,xy=fields
    def d(kind,v):return apply_matrix(matrices[kind],ce,v).scale(GAMMA_BITS)
    vx1=vx.plus(d(1,xx).plus(d(2,xy)).scale(MAP_BITS))
    vy1=vy.plus(d(0,xy).plus(d(3,yy)).scale(MAP_BITS))
    momentum=[vx1,vy1,xx,yy,xy]
    q=[d(0,vx1),d(1,vy1),d(3,vx1),d(2,vy1)]
    div=q[0].plus(q[3]);two_mu=0x40d80000
    xx1=xx.plus(div.scale(LAM_BITS).plus(q[0].scale(two_mu)))
    yy1=yy.plus(div.scale(LAM_BITS).plus(q[3].scale(two_mu)))
    xy1=xy.plus(q[1].plus(q[2]).scale(MU_BITS))
    stress=[vx1,vy1,xx1,yy1,xy1]
    seed=[0]*len(vx.numerator);sn,se=parts(source);seed[source_index]=sn
    source_vector=Vector(seed,se)
    finished=[vx1,vy1,xx1.plus(source_vector),yy1.plus(source_vector),xy1]
    return [momentum,momentum,stress,finished],q

def witness_step(fields,nx,ny,source_index,source):
    """Scalar declared DAG evaluated over independent signed incidence offsets.
    This is an arithmetic witness, separately checked against the exact map.
    """
    size=nx*ny;vx,vy,xx,yy,xy=fields
    one=0x3f800000;zero=0
    def derivative(field,p,axis,forward):
        i,j=p%nx,p//nx;terms=[]
        for m,c in enumerate(COEFFICIENT_BITS,1):
            plus,minus=(m,1-m) if forward else (m-1,-m)
            coords=((i+plus,j),(i+minus,j)) if axis==0 else ((i,j+plus),(i,j+minus))
            vals=[field[y*nx+x] if 0<=x<nx and 0<=y<ny else 0 for x,y in coords]
            terms.append(op('mul',c,op('sub',vals[0],vals[1])))
        result=terms[0]
        for t in terms[1:]:result=op('add',result,t)
        return result
    def corrected(field,p,axis,forward):
        q=op('mul',GAMMA_BITS,derivative(field,p,axis,forward))
        u=op('mul',one,zero);w=op('mul',zero,q);n=op('add',u,w)
        z=op('div',q,one);return op('add',z,n)
    vx1=[];vy1=[]
    for p in range(size):
        a=corrected(xx,p,0,True);b=corrected(xy,p,1,False)
        c=corrected(xy,p,0,False);d=corrected(yy,p,1,True)
        vx1.append(op('add',vx[p],op('mul',MAP_BITS,op('add',a,b))))
        vy1.append(op('add',vy[p],op('mul',MAP_BITS,op('add',c,d))))
    q=[[corrected(f,p,axis,forward) for p in range(size)] for f,axis,forward in
       ((vx1,0,False),(vy1,0,True),(vx1,1,True),(vy1,1,False))]
    xx1=[];yy1=[];xy1=[]
    for p in range(size):
        div=op('add',q[0][p],q[3][p]);two_mu=op('mul',0x40000000,MU_BITS)
        lt=op('mul',LAM_BITS,div)
        xx1.append(op('add',xx[p],op('add',lt,op('mul',two_mu,q[0][p]))))
        yy1.append(op('add',yy[p],op('add',lt,op('mul',two_mu,q[3][p]))))
        xy1.append(op('add',xy[p],op('mul',MU_BITS,op('add',q[1][p],q[2][p]))))
    stress=[vx1,vy1,xx1,yy1,xy1]
    finished=[vx1,vy1,xx1.copy(),yy1.copy(),xy1]
    finished[2][source_index]=op('add',xx1[source_index],source)
    finished[3][source_index]=op('add',yy1[source_index],source)
    momentum=[vx1,vy1,xx,yy,xy]
    return [momentum,momentum,stress,finished],q

def compare(bits,exact):
    """Exact squared norms/residual and simultaneous fixed-input defect bound.
    Returns exact fractions, never a candidate-derived normalization floor.
    """
    native=vectors_from_bits([bits])[0]
    e=min(native.exponent,exact.exponent)
    delta=[(a<<(native.exponent-e))-(b<<(exact.exponent-e)) for a,b in zip(native.numerator,exact.numerator)]
    ds=sum(x*x for x in delta);ns=sum(x*x for x in exact.numerator)
    def scaled(n,exp):return Fraction(n<<exp,1) if exp>=0 else Fraction(n,1<<-exp)
    return scaled(ds,2*e),scaled(ns,2*exact.exponent)

def cross_time_energy(fields,matrices,ce):
    """Exact accepted W invariant, not simultaneous continuum energy.
    C^-1 is derived from frozen lambda=27/4, mu=27/8; H_v is stored M^-1.
    """
    vx,vy,xx,yy,xy=fields
    def d(kind,v):return apply_matrix(matrices[kind],ce,v).scale(GAMMA_BITS)
    q=[d(0,vx),d(1,vy),d(3,vx),d(2,vy)];div=q[0].plus(q[3])
    bx=div.scale(LAM_BITS).plus(q[0].scale(LAM_BITS))
    by=div.scale(LAM_BITS).plus(q[3].scale(LAM_BITS))
    bs=q[1].plus(q[2]).scale(MU_BITS)
    def negate(v):return Vector([-n for n in v.numerator],v.exponent)
    px,py,ps=xx.plus(negate(bx)),yy.plus(negate(by)),xy.plus(negate(bs))
    def dot(a,b):
        n=sum(x*y for x,y in zip(a.numerator,b.numerator));e=a.exponent+b.exponent
        return Fraction(n<<e,1) if e>=0 else Fraction(n,1<<-e)
    return (dot(vx,vx)+dot(vy,vy))/fraction(MAP_BITS)+Fraction(8,81)*(dot(xx,px)+dot(yy,py)) \
        -Fraction(4,81)*(dot(xx,py)+dot(yy,px))+Fraction(8,27)*dot(xy,ps)

def fixtures():
    return [dict(id='B1',nx=13,ny=11,nt=16,source=(6,5),receivers=((3,2),(6,2),(3,5),(6,5))),
            dict(id='B3',nx=48,ny=48,nt=4,source=(24,24),receivers=((20,24),(27,24),(24,20),(24,27),(19,24),(28,24),(24,19),(24,28)))]

def initial_bits(case):
    nx,ny=case['nx'],case['ny'];result=[[0]*(nx*ny) for _ in range(5)]
    if case['id']=='B1':
        for f in range(5):
            for i,j in ((0,0),(nx-1,0),(0,ny-1),(nx-1,ny-1)):
                result[f][j*nx+i]=encode_dyadic((f+1)*(1+i+2*j),-20)
    return result

def build_certificates(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();certificate=[]
    # Qualify scalar software nodes independently with exact rational operands.
    assert round_ratio(1,1<<149)==1
    assert round_ratio(1,1<<150)==0
    assert round_ratio(3,1<<150)==2
    assert round_ratio((1<<24)+1,1<<24)==0x3f800000
    assert round_ratio(1225,1024)==COEFFICIENT_BITS[0]
    assert round_ratio(-245,3072)==COEFFICIENT_BITS[1]
    assert round_ratio(49,5120)==COEFFICIENT_BITS[2]
    assert round_ratio(-5,7168)==COEFFICIENT_BITS[3]
    assert material_maps([MU_BITS]*4,[0x3fc00000]*2)==(MAP_BITS,MU_BITS)
    assert round_ratio(1,1000)==SOURCE_BITS
    baseline_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()
    for case in fixtures():
        COUNTS.clear();nx,ny,nt=case['nx'],case['ny'],case['nt'];n=nx*ny
        matrices,ce=incidence(nx,ny);initial=initial_bits(case)
        exact=vectors_from_bits(initial);rounded=initial
        source_index=case['source'][1]*nx+case['source'][0]
        observations=[j*nx+i for i,j in case['receivers']]
        max_relative=Fraction(0);nonzero=zero=0;max_payload=0
        energy0=cross_time_energy(exact,matrices,ce) if case['id']=='B1' else None
        file=output/(case['id']+'-exact-and-witness.jsonl.gz')
        with gzip.open(file,'xt',encoding='utf-8',newline='\n') as f:
            f.write(json.dumps({'fixture':case,'initial_bits':initial,'coefficient_bits':COEFFICIENT_BITS,
                'gamma':GAMMA_BITS,'map':MAP_BITS,'lambda':LAM_BITS,'mu':MU_BITS,'source_q0':SOURCE_BITS})+'\n')
            for t in range(nt):
                source=SOURCE_BITS if case['id']=='B3' and t==0 else 0
                states,q=exact_step(exact,matrices,ce,source_index,source)
                native,rq=witness_step(rounded,nx,ny,source_index,source)
                rows=[]
                for stage,(a,b) in enumerate(zip(states,native)):
                    for component,(ev,bits) in enumerate(zip(a,b)):
                        error,norm=compare(bits,ev)
                        if norm:
                            nonzero+=1;ratio=error/norm;max_relative=max(max_relative,ratio)
                            assert ratio<=Fraction(1,100000)**2,(case['id'],t,stage,component,'predicted gate unresolved')
                        else:
                            zero+=1
                            # Exact fixed-input arithmetic defect is recorded even
                            # for non-structural zero-reference vectors.
                        rows.append({'stage':stage,'component':component,'exact':ev.payload(),'witness_bits':bits,
                                     'defect_norm2':str(error),'reference_norm2':str(norm)})
                data_exact=[];data_witness=[]
                for p in observations:
                    for c in (0,1):
                        ev=states[0][c]
                        data_exact.append({'n':ev.numerator[p],'e':ev.exponent})
                        data_witness.append(native[0][c][p])
                energy_record=None
                if energy0 is not None:
                    expected_energy=cross_time_energy(states[-1],matrices,ce)
                    assert expected_energy==energy0,'exact finite W invariant defect'
                    witness_energy=cross_time_energy(vectors_from_bits(native[-1]),matrices,ce)
                    energy_record={'intended_exact':str(expected_energy),'initial_exact':str(energy0),
                                   'predicted_native_drift':str(witness_energy-energy0)}
                payload={'time':t,'states':rows,'strain_exact':[x.payload() for x in q],
                         'strain_witness_bits':rq,'data_exact':data_exact,'data_witness_bits':data_witness}
                payload['cross_time_energy_certificate']=energy_record
                # Stream JSON chunks; avoid constructing a huge decimal string.
                for chunk in json.JSONEncoder(separators=(',',':')).iterencode(payload):f.write(chunk)
                f.write('\n')
                max_payload=max(max_payload,sum(sys.getsizeof(x)+sys.getsizeof(x.numerator)+sum(sys.getsizeof(a) for a in x.numerator) for stage in states for x in stage))
                exact=states[-1];rounded=native[-1]
        certificate.append({'fixture':case,'exact_reference_uncertainty':0,'expected_cache_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                            'nonzero_vectors_certified':nonzero,'zero_reference_vectors':zero,'maximum_predicted_relative_norm2':str(max_relative),
                            'operation_counts':dict(COUNTS),'conservative_stage_payload_bytes':max_payload,
                            'measured_all_traced_working_allocations_peak_bytes':tracemalloc.get_traced_memory()[1],
                            'full_process_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    result={'schema':'M9L1A-INDEPENDENT-DYADIC-REFERENCE-01','BASE':'059f65fcdff57fd4465521e76f3a9aa509b69fb7',
            'candidate_outputs_read':False,'source_reference_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'target':'Exact finite A_G/B_G from stored input/coefficient/map/gamma dyadics; rounded DAG separately certified',
            'gates_fixed_before_candidate_outputs':{'composed_relL2':'1/100000','primitive_operand':'32*epsilon32','reference_uncertainty':0},
            'certificates':certificate,'elapsed_seconds':time.monotonic()-start,
            'isolated_interpreter_and_imports_baseline_peak_RSS_KiB':baseline_rss_kib,
            'accounting':'tracemalloc peak includes live case-specific Python allocations, incidence matrices, exact/witness fields, temporary contractions, JSON/gzip buffers and certificates; repeated references are retained and conservatively included. Interpreter/shared-library RSS separately observed.'}
    (output/'reference-certificates.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'reference_certification':'PASS','cases':[{'id':c['fixture']['id'],
        'working_peak_bytes':c['measured_all_traced_working_allocations_peak_bytes'],
        'process_peak_RSS_KiB':c['full_process_peak_RSS_KiB']} for c in certificate],
        'elapsed_seconds':result['elapsed_seconds'],'outputs_read':False}),flush=True)

if __name__=='__main__':build_certificates(sys.argv[1])
