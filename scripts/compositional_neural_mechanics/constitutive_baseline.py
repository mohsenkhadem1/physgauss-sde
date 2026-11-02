"""Post-training analytic control on exactly declared constitutive splits; no fitting."""
import itertools,json
import torch
from scripts.compositional_neural_mechanics.materials import invariants,SharedMaterialLaw
from scripts.compositional_neural_mechanics.train_materials import data,metrics,ROOT,REPORT,OUT,sha,write
class OracleSmallStrainNeoHookean:
 def __call__(self,F,z):
  q1,_,volume=invariants(F);scale=torch.exp(z[...,0]);return 2000*scale*q1+20000*scale*volume
 def stress(self,F,z,create_graph=False):
  F=F.detach().requires_grad_(True);return torch.autograd.grad(self(F,z).sum(),F,create_graph=create_graph)[0]
def main():
 torch.set_num_threads(1);z=list(itertools.product([-.25,.25],[-.5,.5],[-1.,1.]));baseline=OracleSmallStrainNeoHookean()
 result=dict(scope='Oracle matched-small-strain Neo-Hookean baseline, not optimized using test outcomes. True-family generator is the exact analytic ceiling by construction. This is not a scene FEM comparison.',source_sha256=sha(__file__),prototype_result_sha256=sha(REPORT/'constitutive-prototype-results.json'),weight_freeze_sha256=sha(OUT/'weight-freeze.json'))
 for name,strain,seed in [('unseen_descriptor_interpolation',.2,3),('harder_strain_unseen_descriptors',.35,4)]:result[name]=metrics(baseline,data(1024,z,strain,seed,'cpu'))
 write(REPORT/'constitutive-analytic-baseline.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
