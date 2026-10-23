"""Strict codec for ONE already-acquired ideal static probe; zero file/truth I/O."""
import numpy as np
from experiments.campaign_c38_postprobe_inference import PROBE_KEYS,validate_acquired_probe
STRINGS={'information_role','object_id','observation_kind'}
FLAGS={'reset_from_common_rest','is_RGBD_perception','is_temporal_forecast'}


def encode_probe(prefix,probe,*,design):
 validate_acquired_probe(prefix,probe,design=design)
 out={}
 for k,v in probe.items():
  if k in STRINGS:
   if type(v)is not str:raise ValueError('Python string probe scalar required')
   out[k]=np.asarray(v)
  elif k in FLAGS:
   if type(v)is not bool:raise ValueError('Python bool probe scalar required')
   out[k]=np.asarray(v,dtype=np.bool_)
  elif k=='acquisition_index':out[k]=np.asarray(v,dtype=np.int64)
  elif k=='force_Fz_N':out[k]=np.asarray(v,dtype=np.float64)
  else:out[k]=np.asarray(v).copy()
 return out


def decode_probe(prefix,members,*,design):
 if set(members)!=PROBE_KEYS:raise ValueError('Exact sanitized public probe NPZ allowlist required')
 out={}
 for k,v in members.items():
  a=np.asarray(v)
  if k in STRINGS:
   if a.shape!=() or a.dtype.kind!='U':raise ValueError('Scalar Unicode public probe required')
   out[k]=str(a.item())
  elif k in FLAGS:
   if a.shape!=() or a.dtype!=np.bool_:raise ValueError('Scalar bool public probe required')
   out[k]=bool(a.item())
  elif k=='acquisition_index':
   if a.shape!=() or a.dtype!=np.int64:raise ValueError('Scalar int64 acquisition index required')
   out[k]=int(a.item())
  elif k=='force_Fz_N':
   if a.shape!=() or a.dtype!=np.float64:raise ValueError('Scalar float64 public probe force required')
   out[k]=float(a.item())
  else:out[k]=a.copy()
 return validate_acquired_probe(prefix,out,design=design)
