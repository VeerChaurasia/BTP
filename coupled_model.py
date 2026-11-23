"""Supplementary November hydrogel model, prepared 2026-11-22.

Independent Python numerical demonstration, NOT a COMSOL export or a claim of
experimental validation. 2D square fixed-domain hydrogel, water uptake,
hydration-dependent drug diffusion, water-mediated polymer bulk degradation,
and degradation-linked liberation of immobile drug.

Cell-centered finite-volume and implicit Euler implementation. All amounts are
per unit depth of the square 2D domain. Ideal sink for dissolved drug,
unit prescribed surface hydration; spatially uniform dry polymer initially.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import factorized, spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

OUT=Path(__file__).resolve().parent

@dataclass
class Params:
    L: float = 1.0e-3
    C0: float = 1.0
    Dw: float = 4.0e-11
    Ddry: float = 1.0e-12
    Dwet: float = 1.0e-10
    gamma: float = 1.5
    beta: float = 0.5
    kdeg: float = 7.0e-5
    fmobile: float = 0.35
    swelling_coeff: float = 0.6  # postprocessing index only, NOT deformation
    t_end: float = 21600.0
    dt: float = 45.0
    n: int = 31

def diffusion_operator(D,n,dx, boundary_value=0.0):
    """Operator for div(D grad c) with prescribed Dirichlet face values.
    4-neighbor Cartesian FV. Harmonic interior D; half-cell face BC.
    Return A and RHS forcing for prescribed boundary concentration.
    """
    N=n*n; idx=np.arange(N).reshape(n,n)
    d=D.reshape(n,n)
    xface=2*d[:,:-1]*d[:,1:]/(d[:,:-1]+d[:,1:])/(dx*dx)
    yface=2*d[:-1,:]*d[1:,:]/(d[:-1,:]+d[1:,:])/(dx*dx)
    a=np.concatenate([idx[:,:-1].ravel(),idx[:-1,:].ravel()]);
    b=np.concatenate([idx[:,1:].ravel(),idx[1:,:].ravel()]);
    g=np.concatenate([xface.ravel(),yface.ravel()]);
    diag=np.zeros(N); np.add.at(diag,a,g); np.add.at(diag,b,g)
    bd=np.zeros((n,n));bd[:,0]+=2*d[:,0]/(dx*dx);bd[:,-1]+=2*d[:,-1]/(dx*dx)
    bd[0,:]+=2*d[0,:]/(dx*dx);bd[-1,:]+=2*d[-1,:]/(dx*dx)
    bd=bd.ravel();diag+=bd
    rows=np.concatenate([a,b,np.arange(N)]);cols=np.concatenate([b,a,np.arange(N)])
    vals=np.concatenate([g,g,-diag]);
    A=sp.coo_matrix((vals,(rows,cols)),shape=(N,N)).tocsr()
    return A,bd*boundary_value,bd

def simulate(params:Params, snapshots=True):
    p=params;n=p.n;N=n*n;dx=p.L/n;dt=p.dt
    steps=round(p.t_end/dt)
    if abs(steps*dt-p.t_end)>1e-8: raise ValueError('dt must evenly divide t_end')
    I=sp.eye(N,format='csc')
    Awater,water_bc,_=diffusion_operator(np.full(N,p.Dw),n,dx,1.0)
    water_solve=factorized(I-dt*Awater.tocsc())
    h=np.zeros(N);poly=np.ones(N)
    mobile=np.full(N,p.fmobile*p.C0)
    trapped=np.full(N,(1-p.fmobile)*p.C0)
    initial=np.sum(mobile+trapped)*dx*dx
    cumulative_flux=0.0
    records=[];maps={}
    def record(t):
        remaining=float(np.sum(mobile+trapped)*dx*dx)
        drugrelease=100*(1-remaining/initial)
        rec={'time_s':float(t),'time_h':t/3600,'release_pct':drugrelease,
             'mean_hydration':float(np.mean(h)),
             'polymer_remaining_pct':100*float(np.mean(poly)),
             'polymer_loss_pct':100*(1-float(np.mean(poly))),
             'mobile_drug_pct':100*float(np.sum(mobile)*dx*dx/initial),
             'immobile_drug_pct':100*float(np.sum(trapped)*dx*dx/initial),
             'swelling_index_proxy':1+p.swelling_coeff*float(np.mean(h)),
             'cumulative_boundary_loss_pct':100*cumulative_flux/initial,
             'relative_mass_balance_error':abs(initial-remaining-cumulative_flux)/initial}
        records.append(rec)
    record(0.)
    desired={1800,7200,21600}
    for k in range(1,steps+1):
        t=k*dt
        h=water_solve(h+dt*water_bc)
        h=np.clip(h,0,1)
        decay=np.exp(-p.kdeg*h*dt)
        poly=poly*decay
        new_trapped=trapped*decay
        source=trapped-new_trapped
        trapped=new_trapped
        D=p.Ddry+(p.Dwet-p.Ddry)*(h**p.gamma)*(1+p.beta*(1-poly))
        Ad,_,boundary_loss=diffusion_operator(D,n,dx,0.)
        mobile=spsolve(I-dt*Ad.tocsc(),mobile+source)
        cumulative_flux+=dt*float(np.dot(boundary_loss,mobile))*dx*dx
        if k%max(1,int(round(450/dt)))==0 or k==steps:
            record(t)
        if snapshots and any(abs(t-v)<dt/2 for v in desired):
            maps[int(round(t))]={'hydration':h.reshape(n,n).copy(),
                                 'mobile':mobile.reshape(n,n).copy(),
                                 'polymer':poly.reshape(n,n).copy()}
    data=pd.DataFrame(records)
    return data,maps

def analytic_2d_release(t,L,D,terms=250):
    if t==0:return 0.
    i=np.arange(1,2*terms,2,dtype=float)
    s=np.sum(8/(np.pi*np.pi*i*i)*np.exp(-np.pi*np.pi*i*i*D*t/L**2))
    return 100*(1-s*s)

def linear_crossing(t,y,threshold):
    if np.max(y)<threshold:return None
    idx=np.argmax(y>=threshold)
    if idx==0:return float(t[0])
    return float(np.interp(threshold,y[idx-1:idx+1],t[idx-1:idx+1]))

def run():
    p=Params();p_dict=p.__dict__.copy()
    base,maps=simulate(p)
    base.to_csv(OUT/'november_coupled_time_series.csv',index=False)
    results={'base':{'parameters':p_dict,'release_1h_pct':float(np.interp(3600,base.time_s,base.release_pct)),
          'release_6h_pct':float(base.release_pct.iloc[-1]),
          'hydration_6h':float(base.mean_hydration.iloc[-1]),
          'polymer_remaining_6h_pct':float(base.polymer_remaining_pct.iloc[-1]),
          'mass_balance_max_rel_error':float(base.relative_mass_balance_error.max()),
          't50_h':linear_crossing(base.time_h.values,base.release_pct.values,50),
          't90_h':linear_crossing(base.time_h.values,base.release_pct.values,90)}}
    cases={
        'No degradation / immobilized fraction':replace(p,kdeg=0),
        'Half degradation rate':replace(p,kdeg=p.kdeg/2),
        'Base':p,
        'Double degradation rate':replace(p,kdeg=p.kdeg*2),
        'Half water uptake diffusivity':replace(p,Dw=p.Dw/2),
        'Double water uptake diffusivity':replace(p,Dw=p.Dw*2),
        'Half wet-state drug diffusivity':replace(p,Dwet=p.Dwet/2),
        'Double wet-state drug diffusivity':replace(p,Dwet=p.Dwet*2),
    }
    sweeps={};sens=[]
    for name,v in cases.items():
        if name=='Base':dat=base
        else: dat,_=simulate(v,snapshots=False)
        sweeps[name]=dat
        sens.append({'scenario':name,'Dw_m2_s':v.Dw,'Dwet_m2_s':v.Dwet,'kdeg_s_inv':v.kdeg,
                     'R_1h_pct':float(np.interp(3600,dat.time_s,dat.release_pct)),
                     'R_6h_pct':float(dat.release_pct.iloc[-1]),
                     'hydration_6h':float(dat.mean_hydration.iloc[-1]),
                     'polymer_loss_6h_pct':float(dat.polymer_loss_pct.iloc[-1]),
                     't50_h':linear_crossing(dat.time_h.values,dat.release_pct.values,50),
                     't90_h':linear_crossing(dat.time_h.values,dat.release_pct.values,90)})
    sens_df=pd.DataFrame(sens);sens_df.to_csv(OUT/'november_sensitivity.csv',index=False)
    results['sensitivity']=sens
    # Independent convergence checks using same simple solver discretization.
    mesh_checks=[]
    for n in [21,31,41]:
        q=replace(p,n=n)
        dat=base if n==31 else simulate(q,snapshots=False)[0]
        mesh_checks.append({'n':n,'dt_s':q.dt,'R_6h_pct':float(dat.release_pct.iloc[-1]),
            'polymer_loss_6h_pct':float(dat.polymer_loss_pct.iloc[-1]),
            'max_mass_balance_error':float(dat.relative_mass_balance_error.max())})
    dt_checks=[]
    for dt in [90,45,22.5]:
        q=replace(p,dt=dt)
        dat=base if dt==45 else simulate(q,snapshots=False)[0]
        dt_checks.append({'n':q.n,'dt_s':dt,'R_6h_pct':float(dat.release_pct.iloc[-1]),
            'polymer_loss_6h_pct':float(dat.polymer_loss_pct.iloc[-1]),
            'max_mass_balance_error':float(dat.relative_mass_balance_error.max())})
    conv=pd.DataFrame([{'type':'spatial',**v} for v in mesh_checks]+[{'type':'temporal',**v} for v in dt_checks])
    conv.to_csv(OUT/'november_convergence.csv',index=False)
    results['mesh_convergence']=mesh_checks;results['time_convergence']=dt_checks
    # Analytical reference limit of the coupled framework:
    # no immobile drug, no degradation, Ddry = Dwet = constant. Hydration remains decoupled.
    refp=replace(p,fmobile=1.,kdeg=0,Ddry=p.Dwet,Dwet=p.Dwet,beta=0,gamma=1)
    ref,_=simulate(refp,snapshots=False)
    tt=[900,1800,3600]
    results['analytical_limit']=[{'time_s':t,
            'numerical_R_pct':float(np.interp(t,ref.time_s,ref.release_pct)),
            'analytical_R_pct':float(analytic_2d_release(t,p.L,p.Dwet))} for t in tt]

    # Plots using plain Matplotlib defaults. All figures original numerical outputs.
    ext=np.array([0,p.L*1e3,0,p.L*1e3])
    fig,axs=plt.subplots(3,3,figsize=(11.6,9.2),constrained_layout=True)
    for j,(t,vals) in enumerate(maps.items()):
        for i,(state,vmax,ttl,cmap) in enumerate([('hydration',1.0,'Water content h (0-1)','Blues'),
                    ('mobile',p.C0,'Mobile drug C (mol/m³)','viridis'),
                    ('polymer',1.0,'Polymer fraction P/P₀','magma')]):
            ax=axs[i,j];m=ax.imshow(vals[state],origin='lower',extent=ext,vmin=0,vmax=vmax,cmap=cmap)
            ax.set_title(f'{ttl}\nt = {t/3600:g} h',fontsize=9)
            ax.set_xlabel('x (mm)',fontsize=8);ax.set_ylabel('y (mm)',fontsize=8)
            fig.colorbar(m,ax=ax,shrink=.73)
    fig.suptitle('November extension — independently calculated 2D hydration, mobile drug and polymer fields',fontsize=12)
    fig.savefig(OUT/'november_2d_coupled_fields.png',dpi=190,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11.0,7.8),constrained_layout=True)
    axs[0,0].plot(base.time_h,base.release_pct,label='Released drug');axs[0,0].plot(base.time_h,base.mobile_drug_pct,label='Mobile still in gel');axs[0,0].plot(base.time_h,base.immobile_drug_pct,label='Immobile still in gel');axs[0,0].set_ylabel('Initial drug mass (%)');axs[0,0].legend(fontsize=8)
    axs[0,1].plot(base.time_h,base.mean_hydration,label='Mean hydration h');axs[0,1].plot(base.time_h,base.swelling_index_proxy,label='Index 1 + 0.6 mean(h)');axs[0,1].set_ylabel('Hydration / indicative index');axs[0,1].legend(fontsize=8)
    axs[1,0].plot(base.time_h,base.polymer_remaining_pct,label='Polymer remaining');axs[1,0].plot(base.time_h,base.polymer_loss_pct,label='Polymer degraded');axs[1,0].set_ylabel('Initial polymer mass (%)');axs[1,0].legend(fontsize=8)
    axs[1,1].plot(base.time_h,base.release_pct,label='From remaining drug');axs[1,1].plot(base.time_h,base.cumulative_boundary_loss_pct,'--',label='Integrated outward flux');axs[1,1].set_ylabel('Released initial drug (%)');axs[1,1].legend(fontsize=8)
    for ax in axs.flat: ax.set_xlabel('Time (hours)');ax.grid(alpha=.25);ax.set_xlim(0,6)
    fig.suptitle('November extension — hydration-linked release, polymer loss, and mass accounting',fontsize=12)
    fig.savefig(OUT/'november_time_series.png',dpi=190,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12.4,3.8),constrained_layout=True)
    names=[['Half degradation rate','Base','Double degradation rate'],
          ['Half water uptake diffusivity','Base','Double water uptake diffusivity'],
          ['Half wet-state drug diffusivity','Base','Double wet-state drug diffusivity']]
    titles=['Bulk degradation rate kdeg','Water uptake diffusivity Dw','Wet drug diffusivity Dwet']
    for ax,group,ttl in zip(axs,names,titles):
        for nm in group: ax.plot(sweeps[nm].time_h,sweeps[nm].release_pct,label=nm)
        ax.set_title(ttl,fontsize=10);ax.set_xlabel('Time (hours)');ax.set_ylabel('Released drug (%)')
        ax.set_xlim(0,6);ax.set_ylim(0,100);ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle('November extension — one-factor-at-a-time sensitivity (illustrative properties)',fontsize=11)
    fig.savefig(OUT/'november_parameter_sensitivity.png',dpi=190,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10.9,3.7),constrained_layout=True)
    for v in mesh_checks:axs[0].plot(v['n'],v['R_6h_pct'],'o',color='C0')
    axs[0].plot([v['n'] for v in mesh_checks],[v['R_6h_pct'] for v in mesh_checks],'-',color='C0')
    axs[0].set_xlabel('Cells along each side (N × N)');axs[0].set_ylabel('6-h drug release (%)');axs[0].set_title('Spatial discretization sensitivity')
    axs[1].plot([v['dt_s'] for v in dt_checks],[v['R_6h_pct'] for v in dt_checks],'o-',color='C1')
    axs[1].set_xlabel('Timestep (s)');axs[1].set_ylabel('6-h drug release (%)');axs[1].set_title('Implicit Euler timestep sensitivity')
    for ax in axs:ax.grid(alpha=.25)
    fig.suptitle('November extension — discretization checks (not experimental validation)',fontsize=11)
    fig.savefig(OUT/'november_numerical_checks.png',dpi=190,bbox_inches='tight');plt.close(fig)
    (OUT/'november_metrics.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({'base':results['base'],'sensitivity':sens,'spatial':mesh_checks,'temporal':dt_checks,'analytical_limit':results['analytical_limit']},indent=2))

if __name__=='__main__':run()
