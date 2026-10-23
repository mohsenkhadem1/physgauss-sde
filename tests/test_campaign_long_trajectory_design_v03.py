"""CPU-only independent exact-moment assembly and guard diagnosis; no FE solve."""
from fractions import Fraction
from math import factorial
import ast,copy
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from experiments.campaign_long_trajectory_design_v03 import mass_symmetry_guard,full_mass_momentum_ledger,MAX_ASSEMBLY_TERMS_PER_ENTRY
from experiments.campaign_long_trajectory_design import full_mass_momentum_ledger as old_ledger


def exact_reference_Gram():
    basis=[]
    for i in range(4):
        a=[0]*4;a[i]=1;b=[0]*4;b[i]=2;basis.append({tuple(a):Fraction(-1),tuple(b):Fraction(2)})
    for i,j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
        a=[0]*4;a[i]=a[j]=1;basis.append({tuple(a):Fraction(4)})
    G=np.empty((10,10))
    for i,a in enumerate(basis):
        for j,b in enumerate(basis):
            value=Fraction(0)
            for pa,ca in a.items():
                for pb,cb in b.items():
                    p=tuple(x+y for x,y in zip(pa,pb));value+=ca*cb*Fraction(6*np.prod([factorial(x) for x in p]),factorial(3+sum(p)))
            G[i,j]=float(value)
    return G


def test_reject_actual_nonsymmetry_and_nonfinite_without_modifying_mass():
    M=np.array([[2.,.3],[.3,1.]])
    near=M.copy();near[0,1]+=1e-15;before=near.copy();_,row=mass_symmetry_guard(near)
    assert row['maximum_abs_skew_kg']>0 and np.array_equal(near,before)
    bad=M.copy();bad[0,1]+=1e-8
    with pytest.raises(ValueError,match='exceeds justified'):mass_symmetry_guard(bad)
    bad[0,1]=np.nan
    with pytest.raises(ValueError,match='Finite'):mass_symmetry_guard(bad)


@pytest.mark.parametrize('cells',[(32,16,16),(40,20,20)])
def test_actual_assembled_mass_original_unchanged_vs_exact_moment_independent_reference(cells):
    from experiments.campaign_long_supported_block import LongSupportedGeometry
    from experiments.campaign_p2_consistent_transient import scalar_mass
    from experiments.campaign_static_p2_block import EDGES
    assert list(EDGES)==[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    model=LongSupportedGeometry(cells,support='both_ends');M=scalar_mass(model);before=M.data.copy()
    actual_terms=int(np.bincount(model.elements.ravel()).max());assert actual_terms+3<=MAX_ASSEMBLY_TERMS_PER_ENTRY
    G=exact_reference_Gram();T=model.elements;rows=np.repeat(T,10,axis=1).ravel();cols=np.tile(T,(1,10)).ravel();values=(model.density*model.volumes[:,None,None]*G[None]).ravel();keep=rows<=cols
    U=sparse.coo_matrix((values[keep],(rows[keep],cols[keep])),shape=M.shape).tocsr();reference=U+U.T-sparse.diags(U.diagonal())
    assert (reference-reference.T).nnz==0 # independently symmetric reference only, never physical replacement
    _,report=mass_symmetry_guard(M);delta=M-reference
    assert float(abs(delta.data).max())<=report['FP64_pair_accumulation_skew_bound_kg']
    assert report['maximum_abs_skew_kg']>0 # reproduces old guard defect on BOTH actual meshes
    u=np.zeros_like(model.rest);v=np.full_like(u,.001);v[:,1]*=np.sin(model.rest[:,0]);w=v*.9;h=.01;y=u+h*(v+w)/2;F=M@((w-v)/h);R=np.zeros_like(u)
    with pytest.raises(ValueError,match='symmetric'):old_ledger(model.rest,u,v,y,w,M,F,R,h)
    ledger=full_mass_momentum_ledger(model.rest,u,v,y,w,M,F,R,h)
    assert ledger['linear_residual_N_s']<h*1e-8 and ledger['angular_residual_N_m_s']<h*1e-8
    assert ledger['cross_velocity_term_retained_in_original_physical_impulse_residual'] and np.array_equal(M.data,before)
    assert np.isfinite(ledger['evaluated_cross_velocity_angular_impulse_N_m_s'])


def test_physics_worker_AST_is_identical_except_imported_guard_and_runtime_copy_unchanged():
    root=Path(__file__).resolve().parents[1]
    def worker(version):
        tree=ast.parse((root/f'experiments/run_long_smooth_trajectory_{version}.py').read_text())
        value=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        for node in ast.walk(value):
            if isinstance(node,ast.ImportFrom) and node.module=='experiments.campaign_long_trajectory_design_v03':node.module='experiments.campaign_long_trajectory_design'
        return ast.dump(value,include_attributes=False)
    assert worker('v02')==worker('v03')
    assert (root/'experiments/campaign_long_trajectory_runtime_v02.py').read_bytes()==(root/'experiments/campaign_long_trajectory_runtime_v03.py').read_bytes()
