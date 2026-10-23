"""Independent sparse-CPU midpoint reconstruction for NEW capacity controls only."""
import numpy as np
from experiments.campaign_p2_consistent_transient import P2State,advance_consistent


def cpu_midpoint(elastic,mass,mask,u0,v0,h,force,energy_scale_J):
    """Solve midpoint through a half-step BE objective, then reconstruct endpoints.

    This is an algebraic one-step control, not an integrated physical prehistory.
    Full M_cf is retained. No physical damping, altered force or projection.
    """
    half=h/2
    middle,details=advance_consistent(elastic,mass,mask,P2State(u0.copy(),v0.copy(),np.zeros_like(u0)),None,half,force,energy_scale_J,force_tolerance_N=1e-10,maximum_iterations=25)
    um=middle.u;du=2*(um-u0);u1=u0+du;v1=2*middle.v-v0
    E0,_,_,_=elastic(u0);Em,gm,_,Jm=elastic(um);E1,_,_,J1=elastic(u1)
    acceleration=(v1-v0)/h;r=mass@acceleration+gm-force
    K0=.5*np.sum(v0*(mass@v0));K1=.5*np.sum(v1*(mass@v1));work=np.sum(force*du);defect=E1-E0-np.sum(gm*du);stationarity=np.sum(r*du);balance=E1-E0+K1-K0-work
    row=dict(original_objective_free_R_N=details['objective_free_R_N'],original_physical_free_R_N=float(np.linalg.norm(r[~mask])),minimum_midpoint_J=float(Jm.min()),minimum_endpoint_J=float(J1.min()),external_midpoint_work_J=float(work),nonlinear_midpoint_internal_energy_defect_J=float(defect),stationarity_work_J=float(stationarity),total_energy_minus_midpoint_work_J=float(balance),ledger_identity_defect_J=float(balance-defect-stationarity),fixed_reaction_resultant_N=np.where(mask,r,0).sum(0).tolist(),Newton_iterations=details['iterations'])
    if max(row['original_objective_free_R_N']/1e-10,row['original_physical_free_R_N']/1e-9)>1 or min(row['minimum_midpoint_J'],row['minimum_endpoint_J'])<=.1 or np.any(u1[mask]!=0) or np.any(v1[mask]!=0):raise RuntimeError('Independent original midpoint residual/J/constraint gate')
    return dict(u=u1,v=v1,midpoint_u=um,ledger=row)
