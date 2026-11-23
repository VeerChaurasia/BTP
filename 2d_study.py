"""Independent reference calculations for a BTP report. NOT COMSOL outputs.
2D square hydrogel sink model (eigenfunction method), non-sink conservative
finite-volume model, and swelling / erosion-mediated-liberation surrogate.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import factorized
import csv, json, time
ASSETS=Path('/mnt/data/btp_work/assets'); OUT=Path('/mnt/data/btp_work/2d_data'); OUT.mkdir(exist_ok=True)
L=1e-3; Dg=1e-10; Dm=1e-9; C0=1.0; TMAX=86400
ODDS=np.arange(1,200,2,dtype=float)
W1=8/(np.pi**2*ODDS**2)
LAMBDA=(np.pi*ODDS/L)**2

def survival_1d(theta):
    """theta=integral D dt, so diffusivity units are m²."""
    theta=np.asarray(theta,dtype=float)
    z=np.maximum(theta,0)[...,None]
    s=np.sum(W1*np.exp(-LAMBDA*z),axis=-1)
    return np.where(theta==0,1,s)

def rel_1d(t,D=Dg): return 1-survival_1d(np.asarray(t)*D)
def rel_2d(t,D=Dg): return 1-survival_1d(np.asarray(t)*D)**2

def conc_2d(x,y,t,D=Dg):
    """Exact fixed-square sine series, constant D, zero-concentration edges."""
    modes=ODDS[:40]; coeff=4/(np.pi*modes)
    vx=np.sin(np.pi*x[:,None]*modes[None,:]/L) @ (coeff*np.exp(-modes**2*np.pi**2*D*t/L**2))
    vy=np.sin(np.pi*y[:,None]*modes[None,:]/L) @ coeff
    # product of x and y modal sums, with time exponential split equally
    v=np.sin(np.pi*x[:,None]*modes[None,:]/L) @ (coeff*np.exp(-modes**2*np.pi**2*D*t/(2*L**2)))
    u=np.sin(np.pi*y[:,None]*modes[None,:]/L) @ (coeff*np.exp(-modes**2*np.pi**2*D*t/(2*L**2)))
    return np.maximum(0, C0*np.outer(u,v))

def theta_sw(t,alpha=2.0,tau=3600.0):
    t=np.asarray(t,dtype=float)
    return Dg*((1+alpha)*t-alpha*tau*(1-np.exp(-t/tau)))

def combined_modal(dt=60,alpha=0, k=0, f=1, saved_times=()):
    # Eulerian fixed-square model: mobile C diffuses, bound C becomes mobile as
    # first-order decay.  dCmob/dt=D(t)*laplacian(Cmob)+k*Cbound
    # dCbound/dt=-k*Cbound, Cmob surface=0; Cbound immobile. Not geometric erosion.
    m=ODDS[:85]
    eig=(np.pi/L)**2*(m[:,None]**2+m[None,:]**2)
    weights=(8/(np.pi**2*m*m))[:,None]*(8/(np.pi**2*m*m))[None,:]
    b=np.full_like(eig,f)
    times=np.arange(0,TMAX+dt,dt,dtype=float)
    retained=np.empty_like(times)
    retained[0]=1.0
    maps={}
    for j in range(1,len(times)):
        tmid=(times[j-1]+times[j])/2
        Deff=Dg*(1+alpha*(1-np.exp(-tmid/3600)))
        q=np.exp(-eig*Deff*dt)
        s=(1-f)*k*np.exp(-k*tmid)
        b=b*q + s*(-np.expm1(-eig*Deff*dt))/(eig*Deff)
        retained[j]=(1-f)*np.exp(-k*times[j]) + np.sum(weights*b)
        if any(abs(times[j]-st)<dt/2 for st in saved_times):
            x=np.linspace(0,L,91)
            sn=np.sin(np.pi*np.outer(x/L,m))
            aa=4/(np.pi*m)
            # normalized mobile solution C=C0* sum a_m*a_n*b_mn sinmx sinny
            v=(sn*aa)@b@(sn*aa).T
            maps[int(times[j])]=np.maximum(0,v)
    return times,np.clip(1-retained,0,1),maps

def nonsink_fvm(n=121,dt=300.0):
    ltot=11e-3; dx=ltot/n
    xs=(np.arange(n)+.5)*dx-0.5*ltot
    X,Y=np.meshgrid(xs,xs)
    mask=(np.abs(X)<L/2-1e-12)&(np.abs(Y)<L/2-1e-12)
    k=np.full((n,n),Dm);k[mask]=Dg
    idx=np.arange(n*n).reshape(n,n)
    I=np.concatenate((idx[:,:-1].ravel(),idx[:-1,:].ravel()))
    J=np.concatenate((idx[:,1:].ravel(),idx[1:,:].ravel()))
    di=k.ravel()[I];dj=k.ravel()[J]
    g=(2*di*dj/(di+dj))/dx**2
    rr=np.concatenate([I,J,I,J]);cc=np.concatenate([I,J,J,I]);vv=np.concatenate([-g,-g,g,g])
    A=coo_matrix((vv,(rr,cc)),shape=(n*n,n*n)).tocsc()
    step=factorized(identity(n*n,format='csc')-dt*A)
    c=mask.ravel().astype(float)*C0
    m0=np.sum(c)*dx*dx
    nstep=int(TMAX/dt)
    ts=np.arange(nstep+1)*dt; releases=np.empty(nstep+1);errors=np.empty(nstep+1)
    releases[0]=0.;errors[0]=0.
    wanted=[0,1800,7200,21600,86400]
    snaps={0:c.reshape(n,n).copy()}
    for j in range(1,nstep+1):
        c=step(c)
        releases[j]=1-np.sum(c[mask.ravel()])*dx*dx/m0
        errors[j]=np.sum(c)*dx*dx/m0-1
        if int(ts[j]) in wanted: snaps[int(ts[j])]=c.reshape(n,n).copy()
    return ts,releases,errors,snaps,mask,xs

def t_at_fraction(times,releases,target):
    idx=np.searchsorted(releases,target)
    if idx>=len(times): return None
    if idx==0: return times[0]
    return float(np.interp(target,releases[idx-1:idx+1],times[idx-1:idx+1]))

def save_csv(name, columns, rows):
    with open(OUT/name,'w',newline='') as f:
        w=csv.writer(f);w.writerow(columns);w.writerows(rows)

T=np.arange(0,TMAX+60,60,dtype=float)
R1=rel_1d(T)
R2=rel_2d(T)
Rsw=1-survival_1d(theta_sw(T))**2
print('Non-sink solver start',flush=True)
start=time.time()
Tns,Rns,err,nsnaps,mask,xs=nonsink_fvm()
print('non sink done',round(time.time()-start,1),'s',flush=True)
print('coupled calculations start',flush=True)
Tm,Re,emaps=combined_modal(alpha=0,k=1e-4,f=.35,saved_times=(1800,7200,21600))
_,Rcomb,cmaps=combined_modal(alpha=2,k=1e-4,f=.35,saved_times=(1800,7200,21600))
Rsw1=1-survival_1d(theta_sw(T,alpha=1))**2
Rsw4=1-survival_1d(theta_sw(T,alpha=4))**2
print('coupled done',round(time.time()-start,1),'s',flush=True)

# figure1: 2D sink spatial fields
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11})
fig,axs=plt.subplots(1,3,figsize=(11.2,3.5),sharex=True,sharey=True,layout='constrained')
for a,t in zip(axs,[300,1800,5400]):
    c=conc_2d(np.linspace(0,L,121),np.linspace(0,L,121),t)
    im=a.imshow(c,extent=(0,1,0,1),origin='lower',vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
    a.set_title(f't = {t/60:g} min');a.set_xlabel('x (mm)')
axs[0].set_ylabel('y (mm)');fig.colorbar(im,ax=axs,label='Mobile drug concentration (mol/m³)',shrink=.82)
fig.suptitle('Independent 2D sink model: square hydrogel concentration maps')
fig.savefig(ASSETS/'2d_sink_maps.png',dpi=175,bbox_inches='tight');plt.close(fig)
# figure2 1d vs 2d and diffusivities
fig,axs=plt.subplots(1,2,figsize=(11.3,4),layout='constrained')
axs[0].plot(T/3600,100*R1,label='1D slab, D = 10⁻¹⁰ m²/s',lw=2.3)
axs[0].plot(T/3600,100*R2,label='2D square, D = 10⁻¹⁰ m²/s',lw=2.3)
axs[0].set_xlim(0,6);axs[0].set_ylim(0,101);axs[0].set_xlabel('Time (hours)');axs[0].set_ylabel('Cumulative release (%)');axs[0].legend(fontsize=9);axs[0].grid(alpha=.2)
for D in [1e-11,1e-10,1e-9]:
    axs[1].plot(T/3600,100*rel_2d(T,D),label=f'D = {D:.0e} m²/s',lw=2.1)
axs[1].set_xlim(0,10);axs[1].set_ylim(0,101);axs[1].set_xlabel('Time (hours)');axs[1].set_ylabel('Cumulative release (%)');axs[1].legend(fontsize=9);axs[1].grid(alpha=.2)
fig.suptitle('2D Fickian sink release: dimensionality and diffusivity checks')
fig.savefig(ASSETS/'2d_sink_release.png',dpi=175,bbox_inches='tight');plt.close(fig)
# fig3 non sink
fig,axs=plt.subplots(1,3,figsize=(11,3.9),layout='constrained')
for a,t in zip(axs,[0,7200,86400]):
    v=nsnaps[t]; im=a.imshow(v,extent=(-5.5,5.5,-5.5,5.5),origin='lower',vmin=0,vmax=1,cmap='magma')
    a.set_title(f't = {t/3600:g} h')
    a.set_xlabel('x (mm)');a.set_xlim(-3,3);a.set_ylim(-3,3)
axs[0].set_ylabel('y (mm)');fig.colorbar(im,ax=axs,label='Concentration (mol/m³)',shrink=.8)
fig.suptitle('2D finite-medium model: concentration spreads beyond a 1 × 1 mm² gel')
fig.savefig(ASSETS/'2d_nonsink_maps.png',dpi=175,bbox_inches='tight');plt.close(fig)
# fig4 nonsink vs sink and mass conservation
fig,axs=plt.subplots(1,2,figsize=(11,4.0),layout='constrained')
axs[0].plot(T/3600,100*R2,label='2D ideal sink',lw=2.3)
axs[0].plot(Tns/3600,100*Rns,label='2D finite medium',lw=2.3)
axs[0].axhline(100*(1-1/121),ls='--',lw=1.2,label='Geometric equilibrium: 99.174%')
axs[0].set_xlim(0,24);axs[0].set_ylim(0,101);axs[0].set_xlabel('Time (hours)');axs[0].set_ylabel('Release from gel (%)');axs[0].legend(fontsize=8.5);axs[0].grid(alpha=.2)
axs[1].plot(Tns/3600,err*100,lw=2)
axs[1].set_xlabel('Time (hours)');axs[1].set_ylabel('Global mass-balance error (% of initial)');axs[1].grid(alpha=.25)
fig.suptitle('2D non-sink: release limit and global mass conservation')
fig.savefig(ASSETS/'2d_nonsink_release_mass.png',dpi=175,bbox_inches='tight');plt.close(fig)
# fig5 swelling and erosion
fig,axs=plt.subplots(1,2,figsize=(11.1,4.1),layout='constrained')
for r,label in [(R2,'Diffusion only'),(Rsw,'Swelling proxy (α = 2)'),(Re,'Erosion-linked liberation'),(Rcomb,'Combined swelling + liberation')]:
    axs[0].plot(T/3600,100*r,label=label,lw=2)
axs[0].set_xlim(0,8);axs[0].set_ylim(0,101);axs[0].set_xlabel('Time (hours)');axs[0].set_ylabel('Total drug released (%)');axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
for r,label in [(R2,'α = 0 (no swelling)'),(Rsw1,'α = 1'),(Rsw,'α = 2'),(Rsw4,'α = 4')]:
    axs[1].plot(T/3600,100*r,label=label,lw=2)
axs[1].set_xlim(0,5);axs[1].set_ylim(0,101);axs[1].set_xlabel('Time (hours)');axs[1].set_ylabel('Total drug released (%)');axs[1].legend(fontsize=8);axs[1].grid(alpha=.2)
fig.suptitle('2D parametric study: time-varying diffusivity and erosion-linked release')
fig.savefig(ASSETS/'2d_swelling_erosion.png',dpi=175,bbox_inches='tight');plt.close(fig)
# fig6 coupled concentration maps
fig,axs=plt.subplots(1,3,figsize=(11,3.7),layout='constrained')
for a,t in zip(axs,[1800,7200,21600]):
    im=a.imshow(cmaps[t],extent=(0,1,0,1),origin='lower',vmin=0,vmax=.030,cmap='viridis')
    a.set_title(f't = {t/3600:.1f} h')
    a.set_xlabel('x (mm)')
axs[0].set_ylabel('y (mm)');fig.colorbar(im,ax=axs,label='Mobile drug concentration (mol/m³)',shrink=.8)
fig.suptitle('2D coupled surrogate: mobile drug (65% initially immobilized)')
fig.savefig(ASSETS/'2d_coupled_maps.png',dpi=175,bbox_inches='tight');plt.close(fig)

models={'1D sink':(T,R1),'2D sink':(T,R2),'2D swelling α=2':(T,Rsw),'2D erosion liberation':(T,Re),'2D combined':(T,Rcomb),'2D non-sink':(Tns,Rns)}
metrics={}
for name,(t,r) in models.items():
    metrics[name]={
        'R_at_1h_pct':float(100*np.interp(3600,t,r)),
        'R_at_2h_pct':float(100*np.interp(7200,t,r)),
        'R_at_6h_pct':float(100*np.interp(21600,t,r)),
        'R_at_24h_pct':float(100*np.interp(86400,t,r)),
        't50_s':t_at_fraction(t,r,.5),
        't90_s':t_at_fraction(t,r,.9),
    }
print(json.dumps(metrics,indent=2),flush=True)
print('nonsink mass err',np.max(np.abs(err)),'equilibrium final',Rns[-1], 'limit',1-1/121,flush=True)
with open(OUT/'metrics.json','w') as f:json.dump({'model_metrics':metrics,'global_mass_balance_relative_max':float(np.max(np.abs(err))),'2d_nonsink_equilibrium_geom':1-1/121,'sim_params':{'L_m':L,'Dgel_m2_s':Dg,'Dmedium_m2_s':Dm,'C0_mol_m3':C0,'total_2d_nonsink_width_m':.011,'f_mobile_initial':.35,'k_liberation_s_inv':1e-4,'swelling_alpha':2,'swelling_tau_s':3600,'model_type':'independent Python reference calculation, not COMSOL'}} ,f,indent=2)
save_csv('sink_2d_time_series.csv',['time_s','release_1d_pct','release_2d_pct','release_2d_swelling_pct','release_erosion_liberation_pct','release_combined_pct'],zip(T,100*R1,100*R2,100*Rsw,100*Re,100*Rcomb))
save_csv('non_sink_2d_time_series.csv',['time_s','release_pct','relative_total_mass_error'],zip(Tns,100*Rns,err))
print('Done',flush=True)
