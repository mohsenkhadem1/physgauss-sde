"""Compact development evidence; no reserved hypotheses or action generation."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.physical_refinement_v15.mechanics_pilot import BASE,REPORT,write,sha,ACTIONS
qpath=REPORT/'mechanics-development-qualification.json';q=json.loads(qpath.read_text());quads=[]
for name in ('homogeneous','gradient_plus','gradient_minus','crossed_support'):
 for j,(action,amp) in enumerate(ACTIONS):
  a=np.load(BASE/'mechanics-pilot'/name/f'n6-q3/action{j}.npz');b=np.load(BASE/'mechanics-pilot'/name/f'n6-q4/action{j}.npz');ta=np.array(q['wrenches'][name]['n6-q3'][j]['torques_Nm']);tb=np.array(q['wrenches'][name]['n6-q4'][j]['torques_Nm']);diff=a['field']-b['field'];ferr=np.linalg.norm(a['reaction_forces_N']-b['reaction_forces_N'],axis=1);terr=np.linalg.norm(ta-tb,axis=1);fr=1e-4+.005*np.linalg.norm(b['reaction_forces_N'],axis=1);tr=1e-5+.005*np.linalg.norm(tb,axis=1);rms=float(np.sqrt(np.mean(diff**2)));mx=float(np.abs(diff).max());quads.append(dict(name=name,action=action,field_rms_mm=rms*1000,field_max_mm=mx*1000,force_difference_N=ferr.tolist(),torque_difference_Nm=terr.tolist(),passed=bool(rms<=2e-5 and mx<=1e-4 and (ferr<=fr).all() and (terr<=tr).all())))
scales=[]
for j,(action,amp) in enumerate(ACTIONS):
 a=np.load(BASE/'mechanics-pilot/homogeneous'/f'n2-q3/action{j}.npz');b=np.load(BASE/'mechanics-pilot/scaled_identifiability_control'/f'n2-q3/action{j}.npz');ta=np.array(q['wrenches']['homogeneous']['n2-q3'][j]['torques_Nm']);tb=np.array(q['wrenches']['scaled_identifiability_control']['n2-q3'][j]['torques_Nm']);scales.append(dict(action=action,field_rms_difference_mm=float(np.sqrt(np.mean((a['field']-b['field'])**2))*1000),force_difference_from_double_N=float(np.linalg.norm(b['reaction_forces_N']-2*a['reaction_forces_N'])),torque_difference_from_double_Nm=float(np.linalg.norm(tb-2*ta))))
pairs=[]
for first,second in [('gradient_plus','gradient_minus'),('homogeneous','crossed_support')]:
 for j,(action,amp) in enumerate(ACTIONS):
  a=np.load(BASE/'mechanics-pilot'/first/f'n6-q3/action{j}.npz');b=np.load(BASE/'mechanics-pilot'/second/f'n6-q3/action{j}.npz');pairs.append(dict(pair=[first,second],action=action,field_coordinate_rms_difference_mm=float(np.sqrt(np.mean((a['field']-b['field'])**2))*1000)))
spatial=[x for x in q['comparisons'] if x['mesh_pair']==[4,6]];path=json.loads((REPORT/'mechanics-path-check.json').read_text());deformations={name:json.loads((BASE/'mechanics-pilot'/name/'n6-q3/complete.json').read_text())['rows'] for name in ['homogeneous','gradient_plus','gradient_minus','crossed_support']}
result=dict(scope='Development-only feasibility, same Q2 Neo-Hookean/material family, gravity0 and physical smooth Robin supports. No test generation, independent-solver truth or uncertainty calibration claim.',candidate_reference=dict(n=6,order=3),spatial_4_vs_6=spatial,quadrature_3_vs_4=quads,all_development_spatial_and_quadrature_gates_pass=all(x['passed'] for x in spatial+quads),continuation=path,scale_invariance_control=scales,development_pair_differences=pairs,deformation=deformations,torque_energy_derivative_error_Nm=q['torque_energy_derivative_max_abs_error_Nm'],qualified_scope='Four specified development hypotheses and three prescribed commands only; new cases/commands require own qualification',sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),qpath,REPORT/'mechanics-path-check.json']})
write(REPORT/'mechanics-pilot-summary.json',result);print(json.dumps(dict(all_gates=result['all_development_spatial_and_quadrature_gates_pass'],max_quad_rms_mm=max(x['field_rms_mm'] for x in quads),scale=scales)))
