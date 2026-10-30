"""Extension-only dynamic body reference qualification; original study unchanged."""
import json
import numpy as np

def inspect(base,object_id,scenario,limit=.00025):
    reasons=[];levels=[]
    for directory,label in [('reference','base'),('reference-refinement','half-dt')]:
        path=base/directory/object_id/scenario
        try:
            status=json.loads((path/'complete.json').read_text())
            if not status['success']:reasons.append(label+' body reference generation failed');continue
            with np.load(path/'trajectory.npz') as a:centers=a['centers'].copy();times=a['times'].copy()
            if centers.ndim!=3 or centers.shape[-1]!=3 or len(centers)!=len(times):reasons.append(label+' body array shape invalid');continue
            if not np.isfinite(centers).all() or not np.isfinite(times).all():reasons.append(label+' body reference nonfinite')
            if len(times)<2 or np.any(np.diff(times)<=0):reasons.append(label+' body time grid invalid')
            if scenario=='moving_sphere' and centers.shape[1]!=3:reasons.append(label+' expected floor/tool/payload identities missing')
            levels.append((centers,times))
        except (OSError,KeyError,ValueError) as e:reasons.append(label+' body reference unavailable: '+str(e))
    rms=None
    if len(levels)!=2:reasons.append('Two body reference levels unavailable')
    else:
        (a,ta),(b,tb)=levels
        if a.shape!=b.shape or ta.shape!=tb.shape or not np.allclose(ta,tb,rtol=0,atol=1e-10):reasons.append('Body reference refinement grid mismatch')
        elif scenario=='moving_sphere':
            rms=float(np.sqrt(np.mean((a[1:,-1]-b[1:,-1])**2)))
            if not np.isfinite(rms) or rms>limit:reasons.append('Dynamic body timestep difference exceeds limit')
    return dict(qualified=not reasons,reasons=reasons,coordinate_rms_m=rms,limit_m=limit,body='Last collider is dynamic payload sphere only in frozen moving_sphere scene')

def augment(quality,base,object_id,scenario,limit=.00025):
    body=inspect(base,object_id,scenario,limit);result=dict(quality)
    result['qualified']=quality['qualified'] and body['qualified'];result['reasons']=list(quality['reasons'])+body['reasons'];result['body_qualification']=body
    return result
