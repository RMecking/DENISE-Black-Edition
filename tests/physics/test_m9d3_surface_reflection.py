"""M9 FD4 forward physics with the existing accepted P/SV reflection gates."""
import ctypes
from dataclasses import replace
import json
import math
import numpy as np
import pytest
from tests.physics.test_m9d3_surface_blocks import surface_blocks,born_library
from tests.physics.test_m9b1_elastic_psv_born_production import Config,F32P,Operator,_ptr
from tests.physics import test_free_surface as accepted
from tests.utilities.elastic_psv_free_surface_reference import fixture
from tests.utilities.seismogram import project_components,absolute_peak_index_in_interval
from tests.utilities.elastic_analytics import two_segment_ray,free_surface_p_coefficients
from tests.utilities.staggered_grid import input_field_position

@pytest.mark.integration
def test_m9_forward_oblique_p_p_and_p_sv(surface_blocks,born_library,tmp_path,monkeypatch):
    surface_blocks.m9d3_run.argtypes=[ctypes.POINTER(Config),F32P]
    traces={}
    def run(directory,*,repository_root,denise_binary,mpiexec,config,nprocx=1,nprocy=1):
        assert (nprocx,nprocy)==(1,1)
        exp=fixture(nx=config.nx,ny=config.ny,nt=config.samples_per_trace,cpml=True,
                    source=(round(config.source_x_m/config.dh_m),round(config.source_y_m/config.dh_m)))
        exp=replace(exp,dh=config.dh_m,dt=config.dt_s,fw=config.absorbing_width_gridpoints,
                    lam=np.full_like(exp.lam,config.density_kg_m3*(config.vp_m_s**2-2*config.vs_m_s**2)),
                    mu=np.full_like(exp.mu,config.density_kg_m3*config.vs_m_s**2),rho=np.full_like(exp.rho,config.density_kg_m3),
                    receivers=tuple((round(x/config.dh_m),round(y/config.dh_m)) for x,y in config.receivers_m),
                    fc=config.source_frequency_hz,source_t0=1.5/config.source_frequency_hz,pml_fpml=config.pml_frequency_hz)
        owner=Operator(born_library,exp,free_surface=int(config.free_surface))
        data=np.empty(owner.data_shape,np.float32)
        try:assert surface_blocks.m9d3_run(ctypes.byref(owner.config),_ptr(data))==0
        finally:owner.close()
        x=(data[:,0,0].astype(float)+data[:,1,0].astype(float))/2
        y=(data[:,0,1].astype(float)+data[:,2,1].astype(float))/2
        traces[config.free_surface]=(config,x,y)
        return data[:,:,0].T.tolist(),data[:,:,1].T.tolist()
    monkeypatch.setattr(accepted,'_run',run)
    accepted.test_psv_free_surface_oblique_modes(tmp_path,None,None,None)
    config,x,y=traces[True];_,cx,cy=traces[False]
    source=input_field_position((config.source_x_m,config.source_y_m),config.dh_m,'sxx')
    receiver=input_field_position((1400.,900.),config.dh_m,'sxy')
    metrics=json.loads((tmp_path/'free_surface_oblique_metrics.json').read_text())
    direct_direction=(receiver[0]-source[0],receiver[1]-source[1])
    direct,_=project_components(x,y,direct_direction)
    direct_peak=1.5/config.source_frequency_hz+math.hypot(*direct_direction)/config.vp_m_s
    direct_index=absolute_peak_index_in_interval(direct,start_s=direct_peak-.05,stop_s=direct_peak+.05,dt_s=config.dt_s)
    signs={};coefficients={}
    for label,speed,transverse in [('P',config.vp_m_s,False),('SV',config.vs_m_s,True)]:
        ray=two_segment_ray(source,receiver,boundary_y_m=5.,incident_velocity_m_s=config.vp_m_s,outgoing_velocity_m_s=speed)
        longitudinal,shear=project_components(x-cx,y-cy,(receiver[0]-ray.boundary_x_m,receiver[1]-5.))
        trace=shear if transverse else longitudinal
        peak=1.5/config.source_frequency_hz+ray.travel_time_s
        index=absolute_peak_index_in_interval(trace,start_s=peak-.05,stop_s=peak+.05,dt_s=config.dt_s)
        signs[label]=float(trace[index])
        angle=math.atan2(ray.boundary_x_m-source[0],source[1]-5.)
        coefficient=free_surface_p_coefficients(angle,vp_m_s=config.vp_m_s,vs_m_s=config.vs_m_s,density_kg_m3=config.density_kg_m3)
        coefficients[label]=coefficient['reflected_sv_displacement' if transverse else 'reflected_p_displacement']
        assert np.sign(signs[label]/direct[direct_index])==np.sign(coefficients[label])
    # With positive explosive sxx/syy, the outgoing P/SV ray convention in the
    # accepted analytic projection fixes the signs; no amplitude fit is used.
    metrics['production_signed_peaks']=signs
    metrics['direct_signed_peak']=float(direct[direct_index])
    metrics['analytic_polarity_coefficients']=coefficients
    (tmp_path/'m9_reflection.json').write_text(json.dumps(metrics,indent=2))
    print('FP32/FP64_PRODUCTION_OBLIQUE_REFLECTION',json.dumps(metrics))
