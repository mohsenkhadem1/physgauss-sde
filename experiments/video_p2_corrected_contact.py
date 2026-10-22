"""Recorded true-P2 drop preview; no simulated motion interpolation or prediction."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from experiments.campaign_p2_contact_root_integration import triangle_shapes
from experiments.campaign_static_p2_block import shapes
from experiments.campaign_p2_keast_quadrature import positive_keast24
from experiments.video_block_gravity_suite import draw_panel
ROOT=Path(__file__).resolve().parents[1]
OP=ROOT/'runs/operator-dev/p2-corrected-contact-video-v01'
OUT=ROOT/'runs/gaussian-campaign-v01/p2-corrected-contact-video-v01'
BASE=ROOT/'runs/operator-dev/p2-corrected-contact-reference-v01'
CAPTURE=ROOT/'runs/gaussian-campaign-v01/p2-corrected-contact-reference-v01/accuracy_1e10/trajectory.npz'
CAPTURE_SHA='633ba220edf8f4d5cfb87adb7e6e7db54a74a6bd321270b4e2e454801c314d1c'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

def tessellation(subdivisions=6):
 if not isinstance(subdivisions,int) or subdivisions<1:raise ValueError('Positive integer subdivisions')
 n=subdivisions;coords=[(i,j) for i in range(n+1) for j in range(n+1-i)];index={p:k for k,p in enumerate(coords)};tri=[]
 for i in range(n):
  for j in range(n-i):
   tri.append([index[i,j],index[i+1,j],index[i,j+1]])
   if i+j<=n-2:tri.append([index[i+1,j],index[i+1,j+1],index[i,j+1]])
 r,s=np.asarray(coords,dtype=float).T/n
 return triangle_shapes(r,s),np.asarray(tri,dtype=int)

def surface_points(q,faces,N,triangles):
 p=np.einsum('pk,fkc->fpc',N,q[faces]);t=(triangles[None]+np.arange(len(faces))[:,None,None]*len(N)).reshape(-1,3)
 return p.reshape(-1,3),t

def positive_volume_samples(rest,elements):
 b,w=positive_keast24();N=shapes(b);X=rest[elements[:,:4]];V=np.linalg.det(np.swapaxes(X[:,1:]-X[:,:1],1,2))/6
 if np.any(V<=0) or np.any(w<=0):raise ValueError('Positive compatible volume weights required')
 weights=(V[:,None]*w[None]).reshape(-1);weights/=weights.sum()
 return N,weights

def rigid_removed(rest,q,elements,N,weights):
 A=np.einsum('pk,fkc->fpc',N,rest[elements]).reshape(-1,3);B=np.einsum('pk,fkc->fpc',N,q[elements]).reshape(-1,3)
 ca=weights@A;cb=weights@B;U,_,Vt=np.linalg.svd((A-ca).T@(weights[:,None]*(B-cb)));fix=np.eye(3);fix[-1,-1]=np.linalg.det(U@Vt);R=U@fix@Vt
 aligned=(q-cb)@R.T+ca;C=(B-cb)@R.T+ca
 return aligned,float(np.sqrt(np.sum(weights[:,None]*(C-A)**2)/3))

def nearest_schedule(times,count=240):
 times=np.asarray(times)
 if times.ndim!=1 or len(times)<2 or np.any(np.diff(times)<=0):raise ValueError('Strict accepted time sequence required')
 requested=np.linspace(times[0],times[-1],count);ids=np.asarray([np.argmin(abs(times-t)) for t in requested],dtype=int)
 return requested,ids

def prepare():
 if (OP/'recipe.json').exists():raise FileExistsError('Frozen recipe exists')
 if sha(CAPTURE)!=CAPTURE_SHA:raise ValueError('Qualified tight capture changed')
 summary=json.loads((BASE/'result-summary.json').read_text());terminal=json.loads((BASE/'terminal-verification.json').read_text())
 if not summary['completed'] or not summary['time_reference_qualified']:raise ValueError('Time pair not passed')
 sources=['experiments/video_p2_corrected_contact.py','tests/test_video_p2_corrected_contact.py','experiments/campaign_p2_contact_root_integration.py','experiments/campaign_static_p2_block.py','experiments/campaign_p2_keast_quadrature.py','experiments/video_block_gravity_suite.py']
 records=[BASE/'protocol.json',BASE/'result-summary.json',BASE/'terminal-verification.json',ROOT/'runs/operator-dev/p2-corrected-contact-order32-v01/CPU-result.json',ROOT/'runs/operator-dev/p2-corrected-contact-order128-v01/result-summary.json']
 recipe=dict(schema='actual-corrected-P2-contact-video-v1',status='FROZEN_BEFORE_SELECTED_CAPTURE_ARRAY_READS',source_sha256={f:sha(ROOT/f) for f in sources},provenance_sha256={str(p.relative_to(ROOT)):sha(p) for p in records},capture=dict(path=str(CAPTURE.relative_to(ROOT)),sha256=CAPTURE_SHA),frames=240,fps=30,requested_times='Uniform physical T0=1/12 to .15s; nearest actual accepted state; duplicates allowed; NO motion interpolation',subdivisions_per_T6_edge=6,rigid_fit='Proper Kabsch on positive exact-volume Keast24 samples of full T10 interpolation; no signed P2 nodal-row weights',geometry_scale=1,camera_orbit_degrees=45,camera_tracks_motion=False,panels=['Actual world surface and compliant plane','Same actual state after proper rigid motion removal'],poster='Largest positive-volume nonrigid RMS among the240renderedstates; visualselectiononly',qualification='Fixed96tet temporal pair passed; fixed-field64/128/adaptive passed; trajectory surface-order and spatial resolution pending; finite compliant plane, NOT rigid floor',is_prediction=False,output=str(OUT.relative_to(ROOT)),CPU_affinity_cores=[30,31])
 write(OP/'recipe.json',recipe);return sha(OP/'recipe.json')

def render(digest):
 p=json.loads((OP/'recipe.json').read_text())
 if sha(OP/'recipe.json')!=digest or sha(CAPTURE)!=CAPTURE_SHA or any(sha(ROOT/f)!=h for f,h in {**p['source_sha256'],**p['provenance_sha256']}.items()):raise ValueError('Changed frozen provenance')
 if set(os.sched_getaffinity(0))!={30,31}:raise ValueError('CPU30/31 only')
 OUT.mkdir(parents=True,exist_ok=False)
 with np.load(CAPTURE,allow_pickle=False) as z:
  times=z['accepted_times_s'];Y=z['accepted_states'];rest=z['rest_m'];elements=z['elements'];faces=z['surface_elements'];floor=float(z['floor_z_m']);requested,ids=nearest_schedule(times,p['frames']);selected=rest[None]+Y[:rest.size,ids].T.reshape(len(ids),*rest.shape)
 if selected.shape!=(240,225,3) or not np.isfinite(selected).all():raise ValueError('Finite recorded96tet fields required')
 N,tri=tessellation(p['subdivisions_per_T6_edge']);rv,rt=surface_points(rest,faces,N,tri);VN,weights=positive_volume_samples(rest,elements);size=(1280,660);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 target=rest.mean(0);worldtarget=target+np.array([0,0,-.025]);writer=imageio_ffmpeg.write_frames(str(OUT/'actual-P2-drop.mp4'),size,fps=30,codec='libx264',output_params=['-crf','19','-threads','2','-movflags','+faststart'],macro_block_size=2);writer.send(None);peak=-1.;poster=None;rms=[]
 try:
  for k,(q,i) in enumerate(zip(selected,ids)):
   aligned,value=rigid_removed(rest,q,elements,VN,weights);rms.append(value);world,tf=surface_points(q,faces,N,tri);removed,_=surface_points(aligned,faces,N,tri);canvas=Image.new('RGB',size,'#f6f8fa');draw=ImageDraw.Draw(canvas);phase=k/239
   draw.text((18,12),'ACTUAL P2 MECHANICS | 96 quadratic tetrahedra | E=300 kPa | finite compliant plane',font=font,fill='#142b3b')
   draw.text((18,42),'World motion; camera target fixed from rest geometry',font=font,fill='#142b3b');draw.text((660,42),'Same state after rigid motion removal; true 1x scale',font=font,fill='#142b3b')
   draw_panel(canvas,world,rv,tf,np.array([],dtype=int),worldtarget,phase,(10,72,630,537),floor)
   draw_panel(canvas,removed,rv,tf,np.array([],dtype=int),target,phase,(650,72,1270,537))
   draw.text((18,549),f'Actual t={times[i]:.6f} s | nearest accepted state #{i} | nonrigid coordinate RMS={value*1e3:.3f} mm',font=font,fill='#142b3b')
   draw.text((18,578),'8-second playback (~120x slow) | 45-degree orbit | six-node P2 surface | no motion interpolation',font=small,fill='#344c5a')
   draw.text((18,605),'Not a learned prediction. Fixed-mesh time check passed; trajectory surface-order and spatial resolution pending.',font=small,fill='#344c5a')
   draw.text((18,628),'Frictionless finite-penalty floor (kappa=1e9 N/m^3); penetration is allowed. Geometry is not exaggerated.',font=small,fill='#344c5a')
   writer.send(np.asarray(canvas).tobytes())
   if value>peak:peak=value;poster=canvas.copy();posterindex=k
 finally:writer.close()
 poster.save(OUT/'preview.png');reader=imageio_ffmpeg.read_frames(str(OUT/'actual-P2-drop.mp4'));next(reader);decoded=sum(1 for _ in reader)
 if decoded!=240:raise ValueError('Encoded framecount mismatch')
 result=dict(completed=True,recipe_sha256=digest,source_sha256=p['source_sha256'],capture=p['capture'],frames=240,decoded_frames=decoded,fps=30,playback_duration_s=8.,physical_duration_s=float(times[-1]-times[0]),slow_factor=8./float(times[-1]-times[0]),accepted_indices=ids.tolist(),requested_times_s=requested.tolist(),actual_times_s=times[ids].tolist(),maximum_time_sampling_error_s=float(max(abs(times[ids]-requested))),interpolated_motion=False,geometry_scale=1,quadratic_face_count=len(faces),rendered_triangles=len(rt),positive_volume_samples=len(weights),peak_nonrigid_coordinate_RMS_m=peak,poster_frame_index=posterindex,qualification=p['qualification'],is_prediction=False,artifacts={f:dict(path=str((OUT/f).relative_to(ROOT)),sha256=sha(OUT/f),bytes=(OUT/f).stat().st_size) for f in ['actual-P2-drop.mp4','preview.png']})
 write(OP/'video-metadata.json',result);return dict(completed=True,frames=decoded,peak_nonrigid_coordinate_RMS_m=peak,output=str(OUT.relative_to(ROOT)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','render']);p.add_argument('--sha');a=p.parse_args();print(json.dumps(prepare() if a.action=='prepare' else render(a.sha)))
