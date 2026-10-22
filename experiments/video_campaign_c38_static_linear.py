"""True1x orbital illustration of independently solved static load states.

Nonlinear reference versus sealed supplied-physics linear prediction. No
physical time, state interpolation, inferred labels, RGB-D or forecast claim.
Quadratic boundary faces are tessellated for display, not a sensor renderer.
"""
from pathlib import Path
import argparse,json,hashlib,os,time
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from experiments.video_block_gravity_suite import camera
from experiments import score_campaign_c38_static_linear as score

ROOT=Path(__file__).resolve().parents[1]
CASE_IDS=('Elo__both_ends__down2','Elo__both_ends__gravity','Elo__both_ends__up2')
EDGES=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
SCORE_PROTOCOL='runs/operator-dev/c38-static-linear-scoring-v01/protocol-v02.json'
SCORE_PROTOCOL_SHA='0bbe477a7bee06ec566522f1d037a0bd92e8683def260f97f9a4ecaf68c5a7d2'
SCORE_SUMMARY='runs/gaussian-campaign-v01/c38-static-linear-scoring-v01/summary.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):return dict(path=str(Path(p).resolve().relative_to(ROOT)),sha256=sha(p))
def boundary6(elements,faces):
 e=np.asarray(elements);f=np.asarray(faces)
 if e.ndim!=2 or e.shape[1]!=10 or f.ndim!=2 or f.shape[1] not in (3,6):raise ValueError('P2 tetrahedra and boundary triangles required')
 edges=np.sort(e[:,np.asarray(EDGES)].reshape(-1,2),axis=1);triples=np.unique(np.column_stack([edges,e[:,4:].ravel()]),axis=0)
 if len(np.unique(triples[:,:2],axis=0))!=len(triples):raise ValueError('Inconsistent shared P2 edge nodes')
 lookup={tuple(r[:2]):r[2] for r in triples};corners=f[:,:3]
 mids=np.asarray([[lookup[tuple(sorted((int(a[i]),int(a[j]))))] for i,j in ((0,1),(1,2),(2,0))] for a in corners])
 result=np.column_stack([corners,mids])
 if f.shape[1]==6 and not np.array_equal(result,f):raise ValueError('P2 boundary node ordering mismatch')
 return result


def display_surface(positions,face6):
 """Four display triangles sample the six actual quadratic nodal locations.

 No deformation amplification or inter-equilibrium state interpolation. This
 is piecewise-planar display tessellation, not a certified depth renderer.
 """
 q=np.asarray(positions);f=np.asarray(face6);local=q[f]
 triangles=np.asarray([[0,3,5],[3,1,4],[5,4,2],[3,4,5]])
 return local[:,triangles].reshape(-1,3,3)


def schedule(frames_per_state=60):
 if type(frames_per_state) is not int or frames_per_state<1:raise ValueError('Positive frame repeat count required')
 return np.repeat(np.arange(3),frames_per_state)


def load_inputs():
 p=score.validate(ROOT/SCORE_PROTOCOL,SCORE_PROTOCOL_SHA);s=json.loads((ROOT/SCORE_SUMMARY).read_text())
 if not s.get('completed') or len(s.get('rows',[]))!=28:raise ValueError('Complete sealed28 comparison required')
 pairs=[]
 for cell in CASE_IDS:
  matches=[r for r in p['pairs'] if r['case']['id']==cell and r['cells'][0]==40];metrics=[r for r in s['rows'] if r['case']['id']==cell and r['cells'][0]==40]
  if len(matches)!=1 or len(metrics)!=1:raise ValueError('Exact same-object finest static load sequence required')
  r=matches[0]
  for name in ('target','prediction','basis'):score._checked(r[name])
  if metrics[0]['capture_records']!={k:r[k] for k in ('target','prediction','basis')}:raise ValueError('Scored render capture mismatch')
  with np.load(ROOT/r['basis']['path'],allow_pickle=False) as z:b={k:z[k].copy() for k in ('rest_m','elements','faces','constraint_mask')}
  with np.load(ROOT/r['target']['path'],allow_pickle=False) as z:t=z['positions_m'].copy();X=z['rest_m'].copy()
  with np.load(ROOT/r['prediction']['path'],allow_pickle=False) as z:q=z['positions_m'].copy()
  if not np.array_equal(X,b['rest_m']) or t.shape!=X.shape or q.shape!=X.shape or not np.isfinite(t).all() or not np.isfinite(q).all():raise ValueError('Exact finite geometry/render field match required')
  pairs.append(dict(case=r['case'],basis=b,target=t,prediction=q,records={k:r[k] for k in ('target','prediction','basis')},q63_coordinate_RMSE_m=metrics[0]['q63']['linear']['coordinate_RMSE_m']))
 for r in pairs[1:]:
  for name in ('rest_m','elements','faces','constraint_mask'):
   if not np.array_equal(r['basis'][name],pairs[0]['basis'][name]):raise ValueError('Static sequence must share one object and boundary')
 return pairs


def panel(canvas,triangles,reference_triangles,rest,phase,box):
 x0,y0,x1,y1=box;width,height=x1-x0,y1-y0;target=(rest.min(0)+rest.max(0))/2;origin=np.array([width/2,height/2+10]);scale=width/.31
 uv,depth,view=camera(triangles.reshape(-1,3),target,phase,origin,scale);uv=uv.reshape(-1,3,2);depth=depth.reshape(-1,3)
 if np.any(uv[...,0]<1) or np.any(uv[...,0]>=width-1) or np.any(uv[...,1]<1) or np.any(uv[...,1]>=height-1):raise ValueError('True1x display geometry clipped')
 image=Image.new('RGB',(width,height),'#f6f8fa');draw=ImageDraw.Draw(image)
 normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-30)
 light=np.array([.3,-.5,1.]);light/=np.linalg.norm(light);shade=.58+.42*abs(normal@light)
 stripes=(reference_triangles.mean(1)[:,0]-rest[:,0].min())/np.ptp(rest[:,0]);colors=np.column_stack([48+135*stripes,127+45*stripes,185-95*stripes])*shade[:,None]
 for i in np.argsort(depth.mean(1)):draw.polygon([tuple(a) for a in uv[i]],fill=tuple(np.clip(colors[i],0,255).astype(int)))
 # Reference bounding-box outline is purely geometric context at1x scale.
 lo,hi=rest.min(0),rest.max(0);corners=np.asarray([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])]);ru,_,_=camera(corners,target,phase,origin,scale)
 for i in range(8):
  for j in range(i+1,8):
   if (i^j) in (1,2,4):draw.line([tuple(ru[i]),tuple(ru[j])],fill='#97a4af',width=1)
 canvas.paste(image,(x0,y0))


def render(output):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only renderer requires empty CUDA visibility')
 start=time.monotonic();pairs=load_inputs();out=Path(output).resolve()
 if not out.is_relative_to(ROOT) or out.exists():raise ValueError('Fresh local render destination required')
 out.mkdir(parents=True);b=pairs[0]['basis'];f6=boundary6(b['elements'],b['faces']);reference=display_surface(b['rest_m'],f6)
 actual=[display_surface(r['target'],f6) for r in pairs];pred=[display_surface(r['prediction'],f6) for r in pairs]
 frames=schedule();size=(1280,640);video=out/'nonlinear-vs-linear-static.mp4';font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 writer=imageio_ffmpeg.write_frames(str(video),size,fps=30,codec='libx264',macro_block_size=2,output_params=['-crf','19','-threads','2','-movflags','+faststart']);writer.send(None)
 try:
  for frame,index in enumerate(frames):
   canvas=Image.new('RGB',size,'#f6f8fa');draw=ImageDraw.Draw(canvas);r=pairs[index]
   draw.text((18,10),'Independent STATIC equilibria | same known material/support | true1x geometry',font=font,fill='#142b3b')
   draw.text((18,44),'Nonlinear reference (simulation)',font=font,fill='#142b3b');draw.text((660,44),'Linear prediction (supplied physics)',font=font,fill='#142b3b')
   draw.text((18,75),f"Acquisition {index+1}/3 | Fz={r['case']['Fz_N']:+g} N plus gravity | E=75,000 Pa | both ends fixed",font=small,fill='#344c5a')
   phase=frame/(len(frames)-1)
   panel(canvas,actual[index],reference,b['rest_m'],phase,(10,100,630,530));panel(canvas,pred[index],reference,b['rest_m'],phase,(650,100,1270,530))
   draw.text((18,550),f"q63 coordinate RMSE: {r['q63_coordinate_RMSE_m']*1e6:.2f} um | gray: undeformed outline |45-degree shared orbit",font=font,fill='#142b3b')
   draw.text((18,584),'Static acquisition order ONLY; independent states held, no physical time or forecast; no exaggeration.',font=small,fill='#344c5a')
   draw.text((18,612),'Supplied E/support, not inferred from images. Display tessellation is not a qualified RGB-D sensor render.',font=small,fill='#344c5a')
   writer.send(np.asarray(canvas).tobytes())
   if frame%60==30:canvas.save(out/f'state-{index}.png')
 finally:writer.close()
 reader=imageio_ffmpeg.read_frames(str(video));next(reader);decoded=sum(1 for _ in reader)
 if decoded!=len(frames):raise ValueError('Decoded frame count mismatch')
 info=dict(schema='C38-static-oracle-linear-video-v01',completed=True,frames=decoded,fps=30,duration_s=len(frames)/30,geometric_scale=1,camera_orbit_degrees=45,static_load_sequence_Fz_N=[r['case']['Fz_N'] for r in pairs],no_physical_times=True,no_state_interpolation=True,no_temporal_forecast=True,no_RGBD_sensor_claim=True,supplied_physics_oracle_A=True,is_physical_ground_truth=False,full_surface_accuracy_qualified=False,surface_display='True six-node quadratic boundary locations split into four planar display triangles; no sensor-render accuracy claim',q63_coordinate_RMSE_m=[r['q63_coordinate_RMSE_m'] for r in pairs],captures=[r['records'] for r in pairs],comparison_protocol=rec(ROOT/SCORE_PROTOCOL),comparison_summary=rec(ROOT/SCORE_SUMMARY),renderer=rec(Path(__file__)),camera_primitive=rec(Path(__file__).with_name('video_block_gravity_suite.py')),raw_media_local=True,mp4_sha256=sha(video),seconds=time.monotonic()-start)
 (out/'video.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n');return info
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=render(a.output);print(json.dumps({k:r[k] for k in ('completed','frames','raw_media_local','seconds')}))
