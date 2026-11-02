"""Serialization-only wrapper for the released reference sensitivity analysis.
The original implementation and release digest are unchanged; NumPy scalar
values are converted to their corresponding Python scalar types before JSON.
"""
import argparse
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17 import cohort_reference_sensitivity as original

def normalize(value):
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,dict):return {k:normalize(v)for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [normalize(v)for v in value]
 return value

def main(bindings,release):
 writer=original.write
 def serializable_write(path,result):
  result=normalize(result)
  result['source_sha256'][str(Path(__file__).relative_to(original.ROOT))]=original.sha(__file__)
  result['serialization_note']='Original released analysis unchanged. NumPy scalar Boolean/float values converted to Python equivalents for JSON; no metric or threshold change.'
  writer(path,result)
 original.write=serializable_write
 try:original.main(bindings,str(Path(release).resolve()))
 finally:original.write=writer

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 for key in ('protocol-path','freeze-path','approval-path','protocol-sha','freeze-sha'):parser.add_argument('--'+key,required=True)
 parser.add_argument('--release-path',required=True);args=vars(parser.parse_args());release=args.pop('release_path');main(args,release)
