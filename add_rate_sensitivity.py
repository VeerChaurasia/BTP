import sys,contextlib,io
sys.path.insert(0,'/mnt/data/btp_work')
with contextlib.redirect_stdout(io.StringIO()):
 import importlib
 s=importlib.import_module('2d_study')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json, csv
K=[5e-5,1e-4,2e-4]
series=[]
for k in K:
 t,r,_=s.combined_modal(alpha=0,k=k,f=.35)
 series.append((k,t,r))
for k,t,r in series:print('k',k,'R1h',float(100*__import__('numpy').interp(3600,t,r)),'R6h',float(100*__import__('numpy').interp(21600,t,r)),'t90hrs',s.t_at_fraction(t,r,.9)/3600)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11})
fig,axs=plt.subplots(1,2,figsize=(11.1,3.9),layout='constrained')
for k,t,r in series:
 axs[0].plot(t/3600,100*r,label=f'k = {k:.0e} s⁻¹',lw=2.2)
axs[0].set_xlim(0,12);axs[0].set_ylim(0,101)
axs[0].set_xlabel('Time (hours)');axs[0].set_ylabel('Total released (%)');axs[0].legend(fontsize=9);axs[0].grid(alpha=.2)
ks=[k for k,t,r in series];t90=[s.t_at_fraction(t,r,.9)/3600 for k,t,r in series]
axs[1].plot([k*1e4 for k in ks],t90,marker='o',lw=2.2)
axs[1].set_xlabel('Liberation rate, k (×10⁻⁴ s⁻¹)');axs[1].set_ylabel('Time to 90% release (hours)');axs[1].grid(alpha=.2)
fig.suptitle('Independent 2D erosion-linked liberation rate sensitivity (65% initially immobilized)')
fig.savefig('/mnt/data/btp_work/assets/2d_liberation_sensitivity.png',dpi=175,bbox_inches='tight');plt.close(fig)
with open('/mnt/data/btp_work/2d_data/erosion_rate_sensitivity.csv','w',newline='') as f:
 w=csv.writer(f);w.writerow(['k_s_inv','R_1h_pct','R_6h_pct','t90_h'])
 for k,t,r in series:w.writerow([k,float(100*__import__('numpy').interp(3600,t,r)),float(100*__import__('numpy').interp(21600,t,r)),s.t_at_fraction(t,r,.9)/3600])
