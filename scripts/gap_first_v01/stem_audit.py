"""Independent development qualification; no cohort outcomes or protocol edits."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.special import ndtr
from .stem_mechanics import beam_fem,response_row
from .stem_inference import Posterior,expected_risk

def green(x,a,L,EI,k):
 x=np.asarray(x);small=np.minimum(x,a);large=np.maximum(x,a)
 return 1/k+small**2*(3*large-small)/(6*EI)

def qualify_mechanics(config):
 records=[];F=config['initial_force_n'];h=config['thickness_m'];w=config['width_m'];I=w*h**3/12
 fractions=sorted(set([1.,config['heldout_load_fraction'],*config['push_candidates_fraction']]))
 for L in config['lengths_m']:
  for b in config['compliance_prior_m_per_n']:
   for s in config['compliance_prior_m_per_n']:
    EI=L**3/(3*b);k=1/s
    for a in fractions:
     for ne in [20,40,80]:
      x,u,rotation=beam_fem(L,EI,k,F,a,ne)
      exact=F*green(x,a*L,L,EI,k)
      formula=response_row(x,a*L,L,F)@np.array([b,s])
      assert np.max(abs(exact-formula))<1e-15
      error=float(np.max(abs(u-exact)));assert error<1e-8
      curvature=F*a*L/EI;maxslope=F*(a*L)**2/(2*EI)
      assert abs(np.max(abs(rotation))-maxslope)<1e-7
      # Timoshenko shear estimate for Poisson=.3, kappa=5/6: F*aL/(kappa*G*A).
      E=EI/I;G=E/(2*1.3);shear=F*a*L/((5/6)*G*w*h)
      records.append(dict(length_m=L,bending_compliance=b,support_compliance=s,load_fraction=a,elements=ne,
       max_nodal_displacement_error_m=error,peak_slope_rad=maxslope,max_surface_bending_strain=curvature*h/2,
       modulus_Pa=E,base_stiffness_N_per_m=k,shear_displacement_estimate_m=shear,
       max_displacement_m=float(np.max(abs(exact)))))
 L=.12
 ratios={}
 for name,theta in [('A',[.4,.1]),('B',[.1,.4])]:
  tip=response_row(L,L,L,F)@theta
  ratios[name]=dict(half_look=float(response_row(L/2,L,L,F)@theta/tip),half_push=float(response_row(L/2,L/2,L,F)@theta/tip))
 return dict(passed=True,records=records,record_count=len(records),ratios=ratios,
  max_slope_rad=max(r['peak_slope_rad'] for r in records),max_strain=max(r['max_surface_bending_strain'] for r in records),
  max_nodal_error_m=max(r['max_nodal_displacement_error_m'] for r in records),
  scope='Independent cubic Hermite beam versus exact Green function, not nonlinear3Dvalidation. Shear estimates assumePoisson.3,kappa5/6; geometric rotations remain model approximation.')

def qualify_risk():
 # Two-state model: target ±d, measurement means ±z, Gaussiannoise sigma.
 # For posterior-mean prediction, exactfailure integral is Phi((a-z)/sigma),
 # a=sigma²/(2z)*log((1-r)/r),r=tolerance/(2d).
 d,z,sigma,tol=1.,1.,.4,.5
 p=Posterior(np.array([[-d,-z],[d,z]]),np.array([.5,.5]),np.zeros(2))
 action=dict(row=np.array([0.,1.]),sigma=sigma,cost=.04)
 r=tol/(2*d);threshold=sigma**2/(2*z)*np.log((1-r)/r);exact=float(ndtr((threshold-z)/sigma))
 records=[]
 for bins in [161,321,641,1281]:
  score=expected_risk(p,np.array([1.,0.]),action,tol,bins)
  records.append(dict(bins=bins,expected_failure=score['expected_failure'],absolute_error=abs(score['expected_failure']-exact)))
 assert records[-1]['absolute_error']<2e-4
 uninformative=dict(row=np.zeros(2),sigma=.4,cost=.015)
 no=expected_risk(p,np.array([1.,0.]),uninformative,tol,161)
 assert abs(no['expected_failure']-1)<1e-12
 return dict(passed=True,exact_continuous_failure_probability=exact,quantized_observation_results=records,
 uninformative_risk=no['expected_failure'],scope='Exact2hypothesis Gaussianmeasurement identity; numericalbinconvergence is not stemcohortvalidation. Failure measured againstposterior-mean pointprediction.')

def main():
 configpath=Path(__file__).with_name('stem_protocol.json');config=json.loads(configpath.read_text());start=time.time()
 out=Path('reports/gap-first-v01');out.mkdir(exist_ok=True,parents=True)
 doc=dict(mechanics=qualify_mechanics(config),risk=qualify_risk(),elapsed_seconds=time.time()-start,
  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [configpath,Path(__file__),Path(__file__).with_name('stem_mechanics.py'),Path(__file__).with_name('stem_inference.py')]})
 (out/'stem-independent-qualification.json').write_text(json.dumps(doc,indent=2)+'\n')
 print(json.dumps({k:v for k,v in doc['mechanics'].items() if k!='records'},indent=2));print(json.dumps(doc['risk'],indent=2))
if __name__=='__main__':main()
