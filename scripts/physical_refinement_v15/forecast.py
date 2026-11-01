"""Sequential policy rollout with selected observations only; no target reader."""
import numpy as np
from scripts.physical_refinement_v15.posterior import Likelihood,posterior
from scripts.physical_refinement_v15.belief import choose


def rollout(prior,initial,queries,design,request,*,policy,fixed_sequence=(),seed=0,fantasies=128,budget=2):
    if budget not in (0,1,2):raise ValueError('Declared budgets are zero, one, two')
    names=[q.name for q in queries]
    if len(set(names))!=len(names) or initial.name in names:raise ValueError('Unique query identities required')
    terms=[Likelihood(initial).log_density(request(initial.name))]
    weights,z=posterior(prior,terms);history=[]
    snapshots=[weights.copy()];diagnostics=[];evidence=[z]
    by_name={q.name:q for q in queries}
    for step in range(budget):
        name,info=choose(weights,queries,design,policy,used=history,fixed_sequence=fixed_sequence,
                         seed=seed+step,fantasies=fantasies)
        # Only now ask for this actual measurement. Neither selection nor this
        # function has a handle to unselected arrays or evaluation responses.
        value=request(name)
        terms.append(Likelihood(by_name[name]).log_density(value));history.append(name)
        weights,z=posterior(prior,terms)
        snapshots.append(weights.copy());diagnostics.append(info);evidence.append(z)
    return np.asarray(snapshots),dict(sequence=history,acquisition=diagnostics,log_evidence=evidence,
                                     seed=seed,fantasies=fantasies,budget=budget)
