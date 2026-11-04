"""Frozen CPU-only block push/withdraw qualification; no dataset access."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.unilateral_attachment import UnilateralAttachmentFEM,unilateral_plane
from scripts.external_benchmark.fem import ExternalFEM
from scripts.external_benchmark.data import rigid_map

OUT=Path('reports/gaussian-boundary-flow-v02/unilateral-qualification')
PLAN=dict(scope='Quasistatic supplied-plane mechanical qualification, no impact or dataset inference',
          nodes_per_axis=4,block_side_m=.1,push_m=.006,withdraw_above_initial_m=.020,
          increments_per_phase=[8,16,32],bottom_stiffness=1e5,top_stiffness=1e3,
          gaussian_xy_sigma_m=.035,gaussian_z_sigma_m=.003,gravity=False,floor=False,
          maxiter=100,preferred_solver_residual=1e-9,
          gates=dict(maximum_residual=1e-7,minimum_J=.03,shared_command_refinement_rms_m=.0001,
                     separated_force_norm=1e-10,final_unilateral_return_rms_m=.0001),
          failure_policy='Preserve every solver/gate failure; no tolerance relaxation or omitted worlds')


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    plan={**PLAN,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
          [Path(__file__),Path('scripts/gaussian_boundary_flow_v02/unilateral_attachment.py'),Path('scripts/external_benchmark/fem.py')]}}
    path=OUT/'plan.json'
    if path.exists():raise RuntimeError('Immutable qualification already exists')
    path.write_text(json.dumps(plan,indent=2)+'\n')
    a=np.linspace(0,.1,4);nodes=np.stack(np.meshgrid(a,a,a,indexing='ij'),-1).reshape(-1,3)
    radial=np.exp(-((nodes[:,:2]-.05)**2).sum(1)/(2*.035**2))
    bottom=radial*np.exp(-nodes[:,2]**2/(2*.003**2))
    top=radial*np.exp(-(nodes[:,2]-.1)**2/(2*.003**2))
    records={};arrays={};start=time.time()
    for steps in PLAN['increments_per_phase']:
        displacement=np.r_[np.linspace(0,-.006,steps+1),np.linspace(-.006,.020,steps+1)[1:]]
        targets=np.tile(nodes,(len(displacement),2,1,1));targets[:,1,:,2]+=displacement[:,None]
        data=dict(nodes=nodes,attachment_weights=np.stack([bottom,bottom*0+top]),
                  controller_targets=targets,attachment_active=np.ones((len(displacement),2)),floor=-1.)
        origins=np.zeros((len(displacement),2,3));origins[:,1,2]=.1+displacement
        normals=np.tile([0.,0.,-1.],(len(displacement),2,1))
        for name in ['bilateral','unilateral']:
            model=(ExternalFEM(data,device='cpu',field=False) if name=='bilateral' else
                   UnilateralAttachmentFEM(data,device='cpu',field=False,plane_origins=origins,
                                           plane_normals=normals,unilateral_groups=[False,True]))
            theta=torch.zeros(model.i_field+1,dtype=model.dtype);theta[0]=-torch.inf
            theta[1]=np.log(1e5);theta[2]=np.log(1e3);theta[model.i_floor]=-torch.inf
            state=None;worlds=[];rows=[];key=f'{name}_{steps}'
            record=dict(method=name,increments=steps,completed=False,frames=[],failure=None)
            try:
                for frame in range(len(displacement)):
                    state,_,diag=model.solve(theta,frame,state,tolerance=1e-9,maxiter=100)
                    world=state.numpy()*model.scale+model.origin
                    if not np.isfinite(diag['residual']) or diag['residual']>1e-7 or diag['minJ']<=.03:
                        raise RuntimeError(f'Unchanged gate failed: {diag}')
                    if name=='unilateral':
                        _,r,_=unilateral_plane(state,model.weights[1:2],torch.exp(theta[2:3]),model.active[frame,1:2],model.plane_origins[frame,1:2],model.plane_normals[frame,1:2])
                    else:
                        r=(model.weights[1]*torch.exp(theta[2]))[:,None]*(state-model.targets[frame,1])
                    R,b=rigid_map(nodes,world);aligned=nodes@R+b
                    row=dict(frame=frame,command_m=float(displacement[frame]),residual=diag['residual'],minJ=diag['minJ'],
                             displacement_coordinate_rms_m=float(np.sqrt(np.mean((world-nodes)**2))),
                             rigid_removed_coordinate_rms_m=float(np.sqrt(np.mean((world-aligned)**2))),
                             top_patch_force_norm=float(torch.linalg.vector_norm(r)),
                             force_units='dimensionless energy-gradient units, not Newtons')
                    rows.append(row);worlds.append(world)
                record['completed']=True
            except (RuntimeError,ValueError) as e:record['failure']=str(e)
            record['frames']=rows;records[key]=record
            arrays[key]=np.array(worlds)
            (OUT/f'{key}.json').write_text(json.dumps(record,indent=2)+'\n')
            print(key,record['completed'],record['failure'],flush=True)
    refinements={}
    for name in ['bilateral','unilateral']:
        for coarse,fine in [(8,16),(16,32)]:
            a,b=arrays[f'{name}_{coarse}'],arrays[f'{name}_{fine}']
            valid=records[f'{name}_{coarse}']['completed'] and records[f'{name}_{fine}']['completed']
            delta=float(np.sqrt(np.mean((a-b[::2])**2))) if valid else None
            refinements[f'{name}_{coarse}_{fine}']=dict(shared_command_rms_m=delta,passed=valid and delta<=.0001)
    passed=all(r['completed'] for r in records.values()) and all(r['passed'] for r in refinements.values())
    for steps in PLAN['increments_per_phase']:
        row=records[f'unilateral_{steps}']['frames']
        passed=passed and records[f'unilateral_{steps}']['completed'] and row[-1]['top_patch_force_norm']<=1e-10 and row[-1]['displacement_coordinate_rms_m']<=.0001
    np.savez_compressed(OUT/'trajectories.npz',**arrays)
    summary=dict(plan=plan,passed=bool(passed),records=records,refinements=refinements,seconds=time.time()-start,
                 limits='Known-normal quasistatic block only; no force-scale calibration, friction, impact, contact discovery or dataset gain')
    (OUT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('passed',passed,'seconds',summary['seconds'],flush=True)

if __name__=='__main__':main()
