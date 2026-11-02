"""Two DEVELOPMENT solved-state visual/physical comparisons plus exact-scale control."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from scripts.compositional_neural_mechanics.local_action_pilot import ROOT,REPORT,BASE
from scripts.compositional_neural_mechanics.full_continuous_inference_pilot import build,pack
from scripts.compositional_neural_mechanics.conditional_inference_pilot import POINTS
from scripts.compositional_neural_mechanics.scenes import make_model
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.visual_metric_bridge import render_state,calibrated_cameras,visual_errors,physical_errors
from scripts.compositional_neural_mechanics.dataset_runner import write,npz
from scripts.compositional_neural_mechanics_reporting.legacy_dependency_audit import closure,sha
OUT=BASE/'visual-bridge-development'

def solve(estimator,theta,pose):
 model=make_model(estimator.scene,**estimator.kw);estimator.bind(model,torch.tensor(theta,dtype=torch.float64));u=None
 for fraction in estimator.continuation:
  model.set_controls(pose*fraction);state=model.solve(initial=u,force_tolerance_N=1e-10,maximum_iterations=60);u=state['displacement_local_m']
 w=model.reaction_wrenches(state)
 return model,state,dict(displacement=model.sample_material(u,POINTS),wrenches=np.concatenate([w['forces_N'],w['torques_Nm']],axis=-1),residual_N=float(np.linalg.norm(state['residual'])),minimum_J=float(np.min(state['J'])))

def main():
 torch.set_num_threads(1)
 if OUT.exists():raise ValueError('Preserve earlier pilot')
 OUT.mkdir();inputs=[]
 for i in (0,1):inputs += [REPORT/f'full-continuous-inference-pilot-world{i}.json',BASE/f'local-action-development/world{i}/initial-fit.json',BASE/f'local-action-development/world{i}/fits.json',BASE/f'local-action-development/world{i}/public.json']
 plan=dict(scope=__doc__,cases=[dict(world=0,held_command=0),dict(world=1,held_command=1)],cameras='Fixed front/rear/oblique256x192 focal250; samecamera allmethods',appearance='FixedmaterialcoordinateRGB=.2+.6q; no materiallabel appearance',methods=['analytic_n4_reference','learned_initial_n2','learned_after_stretch_n2'],sources=closure([Path(__file__).resolve()]),inputs={str(p.relative_to(ROOT)):sha(p)for p in inputs},limitations='Syntheticknowngeometry/appearance; sameQ2state drivesimages andmechanics, no neuralimagewarp. Finitepixels/tessellation, not sensorcalibrated. No unknown6inputs/targets.')
 write(REPORT/'visual-bridge-development-plan.json',plan);results=[];sheet=Image.new('RGB',(256*3,220*6),'white');draw=ImageDraw.Draw(sheet)
 for i in (0,1):
  prior=json.loads((REPORT/f'full-continuous-inference-pilot-world{i}.json').read_text());folder=BASE/f'local-action-development/world{i}';public=json.loads((folder/'public.json').read_text());pose=np.asarray(public['held_poses'][i]);geometry=GeometrySpec(**public['geometry']);dummy=[dict(pose=np.zeros((2,6)),points=POINTS,displacement=np.zeros_like(POINTS))]
  initial=json.loads((folder/'initial-fit.json').read_text())['parameters'];after=json.loads((folder/'fits.json').read_text())['stretch']['parameters'];states={};images={};cams=calibrated_cameras(geometry)
  for method,theta,learned,n in [('analytic_n4_reference',pack(prior['scene']),False,4),('learned_initial_n2',initial,True,2),('learned_after_stretch_n2',after,True,2)]:
   estimator=build(geometry,public['categories'],dummy,learned);estimator.kw['n']=n;model,state,metrics=solve(estimator,theta,pose);states[method]=metrics;images[method]=render_state(model,state['displacement_local_m'],cams)
   npz(OUT/f'world{i}-{method}-state.npz',rest=model.rest,elements=model.elements,material_nodes=model.material_coordinates,displacement=state['displacement_local_m'],pose=pose,wrenches=metrics['wrenches'])
   for camera,image in images[method].items():npz(OUT/f'world{i}-{method}-{camera}.npz',rgb=image['rgb'],depth=image['depth'])
  reference=states['analytic_n4_reference']
  for method in plan['methods'][1:]:
   value=states[method];results.append(dict(world=i,method=method,physical=physical_errors(value['displacement'],reference['displacement'],value['wrenches'],reference['wrenches']),views={camera:visual_errors(images[method][camera],images['analytic_n4_reference'][camera])for camera in cams},residual_N=value['residual_N'],minimum_J=value['minimum_J']))
  for r,camera in enumerate(cams):
   for c,method in enumerate(plan['methods']):
    y=(i*3+r)*220;im=Image.fromarray(np.uint8(np.clip(images[method][camera]['rgb'],0,1)*255));sheet.paste(im,(c*256,y+28));draw.text((c*256+4,y+4),f'dev{i} {camera} {method}',fill='black')
 sheet.save(OUT/'visual-physical-contact-sheet.png')
 # Exact common-scale ambiguity: no gravity/other unscaled forces; use same
 # validated equilibrium under twice BOTH bulk and bilateral interface scales.
 scene=copy.deepcopy(json.loads((REPORT/'full-continuous-inference-pilot-world0.json').read_text())['scene']);scene['gravity']=[0.,0.,0.];model=make_model(scene,n=2);pose=np.zeros((2,6));pose[1,0]=.004;model.set_controls(pose);state=model.solve(force_tolerance_N=1e-10);u=state['displacement_local_m']
 scaled=copy.deepcopy(scene);scaled['bulk_latent'][0]+=np.log(2)
 for z in scaled['interface_latents']:z[0]+=np.log(2)
 other=make_model(scaled,n=2);other.set_controls(pose);residual=float(np.linalg.norm(other.evaluate(u,False)['residual']));a=model.reaction_wrenches(state);b=other.reaction_wrenches(state);cams=calibrated_cameras(model.geometry);im1=render_state(model,u,cams);im2=render_state(other,u,cams)
 ambiguity=dict(gravity=[0,0,0],common_stiffness_multiplier=2.,same_equilibrium_scaled_residual_N=residual,images_bit_identical=all(np.array_equal(im1[k]['rgb'],im2[k]['rgb'])and np.array_equal(im1[k]['depth'],im2[k]['depth'])for k in cams),reference_forces_N=a['forces_N'].tolist(),scaled_forces_N=b['forces_N'].tolist(),reference_torques_Nm=a['torques_Nm'].tolist(),scaled_torques_Nm=b['torques_Nm'].tolist(),force_scaling_max_abs_error_N=float(np.max(abs(b['forces_N']-2*a['forces_N']))),torque_scaling_max_abs_error_Nm=float(np.max(abs(b['torques_Nm']-2*a['torques_Nm']))),views={k:visual_errors(im2[k],im1[k])for k in cams},scope='Exactmathematicalco-scalingcontrol, samevalidatedequilibrium reused; not a claim gravity-loadedcaseisambiguous.')
 if residual>1e-8 or not ambiguity['images_bit_identical']:raise ValueError('Exact scale diagnostic failed')
 evidence={str(p.relative_to(ROOT)):sha(p)for p in OUT.iterdir()if p.is_file()};write(REPORT/'visual-bridge-development-results.json',dict(plan_sha256=sha(REPORT/'visual-bridge-development-plan.json'),comparisons=results,scale_ambiguity=ambiguity,evidence_sha256=evidence,scope=plan['limitations']))
if __name__=='__main__':main()
