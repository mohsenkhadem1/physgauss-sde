"""Prospective external force program v03; frozen generic P2 physics unchanged.

Development tuning: v02 normal.10 failed conservative target-n4 spatial budget
for one prior corner. Normal.08 is declared before new response solves; axial
.18 per face and all original material/boundary/accuracy thresholds are retained.
"""
from experiments.campaign_surface_p2_static_v02 import surface_traction,SURFACE_QUADRATURE

NORMAL_FORCE_N=.08
AXIAL_FORCE_N_PER_FACE=.18
NORMAL_WARMSTART_SCALE=.8
CENTERS=(.09,.135,.18,.225,.27)


def qualification_loads(n,*,order=SURFACE_QUADRATURE):
    normal=surface_traction(n,[0,0,-1],NORMAL_FORCE_N,order=order)[0]
    top=surface_traction(n,[1,0,0],AXIAL_FORCE_N_PER_FACE,order=order)[0]
    bottom=surface_traction(n,[1,0,0],AXIAL_FORCE_N_PER_FACE,face='bottom',order=order)[0]
    return dict(normal=normal,paired_axial=top+bottom)


def load_catalog(n,*,order=SURFACE_QUADRATURE):
    result={}
    for c in CENTERS:
        label=f'x{round(c*1000):03d}'
        result['normal_'+label]=surface_traction(n,[0,0,-1],NORMAL_FORCE_N,center_x=c,order=order)[0]
        result['paired_axial_'+label]=sum(surface_traction(n,[1,0,0],AXIAL_FORCE_N_PER_FACE,center_x=c,face=face,order=order)[0]for face in('top','bottom'))
    return result


def design():
    return dict(schema='surface-p2-force-v03',normal_force_N=NORMAL_FORCE_N,axial_force_N_per_face=AXIAL_FORCE_N_PER_FACE,centers_m=list(CENTERS),normal_warmstart_scale=NORMAL_WARMSTART_SCALE,unchanged_generic_physics='campaign_surface_p2_static_v02',development_tuned_on_v02=True,old_normal_10_spatial_failure_preserved=True,all_original_gates_unchanged=True,no_automatic_execution=True)
