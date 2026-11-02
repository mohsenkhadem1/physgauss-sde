"""Read-only numerical diagnostics of the completed native VBD development ladder."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/compositional-dynamics-v17/native-newton-dynamics';REPORT=ROOT/'reports/compositional-dynamics-v17';PLAN=REPORT/'native-newton-dynamics-plan.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def aligned(x,y):
    a=x-x.mean(0);b=y-y.mean(0);u,_,vt=np.linalg.svd(a.T@b);r=u@vt
    if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@vt
    return a@r-b

def diagnostics(data,status):
    ledger=data['ledger'];mass=data['masses'];x=data['positions'];center=np.einsum('n,tnj->tj',mass,x)/mass.sum();energy=ledger[:,1:5].sum(1)
    impact=np.flatnonzero(ledger[:,6]>1e-8);t0=float(ledger[impact[0],0])if len(impact)else None
    after=(data['times_s']>=t0)if t0 is not None else np.zeros(len(center),bool)
    return dict(minimum_J=float(ledger[:,5].min()),max_penetration_m=float(ledger[:,6].max()),energy_initial_J=float(energy[0]),energy_final_J=float(energy[-1]),max_relative_energy_change=float(np.max(abs(energy-energy[0]))/max(abs(energy[0]),1e-12)),contact_time_s=t0,postcontact_max_COM_z_m=float(center[after,2].max())if after.any()else None,contact_impulse_Ns=status['contact_impulse_Ns'],momentum_balance_residual_norm_Ns=float(np.linalg.norm(status['momentum_balance_residual_Ns'])))

def main():
    p=read(PLAN);terminal=read(BASE/'terminal.json');assert terminal['plan_sha256']==sha(PLAN)and len(terminal['rows'])==12
    expected=[j['name']for j in p['jobs']];assert [r['job']['name']for r in terminal['rows']]==expected
    evidence={str(PLAN.relative_to(ROOT)):sha(PLAN),str((BASE/'terminal.json').relative_to(ROOT)):sha(BASE/'terminal.json')};cache={};rows=[]
    for entry in terminal['rows']:
        name=entry['job']['name'];folder=BASE/name;status=read(folder/'complete.json');assert status['plan_sha256']==sha(PLAN)
        evidence[str((folder/'complete.json').relative_to(ROOT))]=sha(folder/'complete.json')
        if not status['complete']:rows.append(dict(name=name,complete=False,error=status.get('error')));continue
        path=folder/'trajectory.npz';assert sha(path)==status['trajectory_sha256']==entry['trajectory.npz_sha256'];evidence[str(path.relative_to(ROOT))]=sha(path)
        with np.load(path)as a:data={k:a[k]for k in a.files}
        cache[name]=data;rows.append(dict(name=name,complete=True,diagnostics=diagnostics(data,status),seconds=status['wall_seconds']))
    comparisons=[]
    for case in ['release','drop']:
        dts=[.0002,.0001,.00005]if case=='release'else[.0005,.00025,.000125]
        fine=f'{case}-time-{dts[-1]}'
        pairs=[(f'{case}-time-{dts[0]}',f'{case}-time-{dts[1]}','time'),(f'{case}-time-{dts[1]}',fine,'time'),(f'{case}-iterations-16',fine,'iterations'),(fine,f'{case}-iterations-64','iterations'),(f'{case}-iterations-64',f'{case}-mesh-8-4-4','mesh_and_native_contact_quadrature')]
        for left,right,kind in pairs:
            if left not in cache or right not in cache:comparisons.append(dict(left=left,right=right,kind=kind,complete=False));continue
            a,b=cache[left],cache[right];assert np.allclose(a['times_s'],b['times_s'],atol=1e-12,rtol=0)
            x=a['surface_frames'].reshape(len(a['times_s']),-1,3);y=b['surface_frames'].reshape(len(b['times_s']),-1,3);shape=np.stack([aligned(xx,yy)for xx,yy in zip(x,y)])
            world=float(np.sqrt(np.mean((x-y)**2)));shape_error=float(np.sqrt(np.mean(shape**2)))
            comparisons.append(dict(left=left,right=right,kind=kind,complete=True,world_coordinate_RMS_m=world,shape_coordinate_RMS_m=shape_error,world_point_vector_RMS_m=float(np.sqrt(3)*world),shape_point_vector_RMS_m=float(np.sqrt(3)*shape_error),world_le_0_25mm=world<=.00025,shape_le_0_25mm=shape_error<=.00025))
    result=dict(scope='Native Newton development convergence only; no paired Q2 superiority or independent truth claim. Drop9.8 differs preservedQ2prototype9.81; release initialization matched. Native P1 lumpedmass and vertexcontact differ fromQ2.',plan_sha256=sha(PLAN),source_sha256=sha(__file__),expected=12,completed=sum(r['complete']for r in rows),rows=rows,comparisons=comparisons,rms_convention='CoordinateRMS over all41/251 savedframes and6x17x17 commonmaterialsurfacepoints; duplicate edges retained equally. Perframe bestproperrigid removal forshape. PointvectorRMS alsoexplicit.',qualified=False,qualification_note='No automaticpromotion: inspectall temporal,iteration,spatial,event,energy,impulse differences, not merely completion. Conservativephysics should not hide native numericaldissipation.',evidence_sha256=evidence)
    (REPORT/'native-newton-dynamics-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(completed=result['completed'],comparisons=comparisons),indent=2))
if __name__=='__main__':main()
