"""Reject initial detections inconsistent with the calibrated multiview object.

Uses frame zero only, before physical fitting. A fixed robust ray intersection
and projected-mask distance test prevents grippers/reflections being treated as
object geometry. Rejected views remain on disk and in the quality report.
"""
import argparse,hashlib,json
from pathlib import Path
import cv2,h5py,numpy as np
from scipy.ndimage import distance_transform_edt
from deform360.processing.episode import load_episode_calibration

ROOT=Path(__file__).resolve().parents[2]

def qualify(obj='046-sponge',episode=0):
    root=ROOT/'runs/real-transfer-v19/Deform360/processed'/obj/f'episode_{episode:04d}'
    K,C=load_episode_calibration(root);records=[];rays=[];origins=[];masks=[]
    for path in sorted(root.glob('*/mask_refined.meta.json')):
        camera=path.parent.name;maskpath=path.with_name('mask_refined.h5')
        with h5py.File(maskpath) as f:mask=f['data'][0].astype(bool)
        yy,xx=np.where(mask)
        if len(xx)<16:continue
        ray=np.linalg.solve(K[camera],np.array([xx.mean(),yy.mean(),1.]));ray=C[camera][:3,:3]@ray;ray/=np.linalg.norm(ray)
        rays.append(ray);origins.append(C[camera][:3,3]);masks.append(mask)
        records.append(dict(camera=camera,initial_area=len(xx),mask_sha256=hashlib.sha256(maskpath.read_bytes()).hexdigest()))
    if len(records)<8:raise ValueError('At least eight initial detections required for multiview qualification')
    rays=np.asarray(rays);origins=np.asarray(origins);A=np.eye(3)[None]-rays[:,:,None]*rays[:,None,:]
    b=np.einsum('nij,nj->ni',A,origins);weights=np.ones(len(rays))
    for _ in range(40):
        center=np.linalg.solve(np.einsum('n,nij->ij',weights,A),np.einsum('n,ni->i',weights,b))
        distances=np.linalg.norm(np.einsum('nij,nj->ni',A,center-origins),axis=1)
        weights=np.minimum(1.,.015/np.maximum(distances,1e-9))
    accepted=[];rejected=[]
    for row,mask in zip(records,masks):
        camera=row['camera'];point=(center-C[camera][:3,3])@C[camera][:3,:3];uv=K[camera]@point;uv=uv[:2]/uv[2]
        h,w=mask.shape;visible=point[2]>0 and 0<=uv[0]<w and 0<=uv[1]<h
        distance=float(distance_transform_edt(~mask)[int(round(min(h-1,uv[1]))),int(round(min(w-1,uv[0])))]) if visible else None
        threshold=max(8.,.1*np.sqrt(row['initial_area']))
        keep=visible and distance<=threshold
        row.update(projected_center_px=uv.tolist(),distance_to_mask_px=distance,threshold_px=float(threshold),accepted=bool(keep),
                   reason='initial-mask geometric agreement' if keep else 'initial detection inconsistent with object center or object outside view')
        (accepted if keep else rejected).append(camera)
    if len(accepted)<8:raise ValueError('Insufficient geometrically consistent initial views')
    result=dict(object=obj,episode=episode,frame=0,center_world_m=center.tolist(),accepted_cameras=accepted,rejected_cameras=rejected,
                views=records,criterion='40 IRLS iterations, 15mm ray-distance Huber scale; projected center within max(8px,0.1 sqrt(mask area)) of initial mask; no future frames or prediction errors used.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target=root/'v19-view-quality.json';target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(accepted=len(accepted),rejected=rejected,center=center.tolist()),indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--object',default='046-sponge');p.add_argument('--episode',type=int,default=0);a=p.parse_args();qualify(a.object,a.episode)
