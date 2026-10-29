"""Residual-certified modified Newton solves for nearby settled configurations.

Reuse a tangent while it remains useful, refresh on loss of descent or every
12 iterations, and fall back to the original nonlinear solver when necessary.
The potential, physical guards, and acceptance residual are unchanged.
"""
import time
import numpy as np
from scipy.sparse.linalg import splu


def solve(model,initial=None,force_tolerance_N=1e-9):
    if initial is None:return model.solve(force_tolerance_N=force_tolerance_N)
    u=np.asarray(initial).copy();state=model.evaluate(u);linear=None;H=None;history=[];refresh=True;assemblies=0
    for iteration in range(60):
        r=state['residual'].ravel();norm=float(np.linalg.norm(r));history.append(dict(iteration=iteration,residual_N=norm,potential_J=state['potential_J']))
        if norm<force_tolerance_N:
            state.update(displacement_local_m=u,iterations=iteration,history=history,tolerance_N=force_tolerance_N,tangent_assemblies=assemblies);return state
        try:
            if refresh or iteration%12==0:
                linear=None;H=model.evaluate(u,True)['tangent'].tocsr();assemblies+=1
                if str(model.engine.device).startswith('cuda'):
                    from scripts.phystwin.gpu_multigrid import GPUMultigrid
                    linear=GPUMultigrid(H,model.rest+u,model.engine.device)
                else:linear=splu(H.tocsc(),permc_spec='MMD_AT_PLUS_A')
                refresh=False
            direction=linear.solve(H,-r)[0] if str(model.engine.device).startswith('cuda') else linear.solve(-r)
            if np.linalg.norm(H@direction+r)>max(1e-12,norm*1e-7):raise RuntimeError('Modified Newton linear residual')
            slope=float(r@direction)
            if not np.isfinite(direction).all() or slope>=0:raise RuntimeError('Modified Newton non-descent')
            accepted=False
            for backtrack in range(30):
                alpha=2.**(-backtrack);trial=u+alpha*direction.reshape(u.shape)
                try:candidate=model.evaluate(trial)
                except ValueError:continue
                floor=128*np.finfo(float).eps*max(abs(state['potential_J']),model.young*model.volumes.sum(),1e-30)
                if candidate['potential_J']<=state['potential_J']+1e-4*alpha*slope or (abs(candidate['potential_J']-state['potential_J'])<=floor and np.linalg.norm(candidate['residual'])<norm):
                    u=trial;state=candidate;accepted=True;break
            if not accepted:refresh=True
        except RuntimeError:
            # A refreshed tangent that still fails is handled by the existing
            # validated continuation/recovery path, not an altered objective.
            if assemblies>=2:break
            refresh=True
    linear=None;H=None;model.clear_linear_cache();return model.solve(initial=u,force_tolerance_N=force_tolerance_N)
