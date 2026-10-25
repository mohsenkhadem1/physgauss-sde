"""Source-only Bayesian decision primitives for unknown support layouts.

A finite positive parameter-quadrature bank represents an ALREADY qualified
prefix posterior. Utility uses only its candidate and held-load predictions;
no true layout, true parameters, observed future poke or held truth is accepted.
Scalar poke measurements enable controlled one-dimensional outcome integration.
This finite-bank control does not qualify the underlying physical/inverse model.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp,ndtr


def _positive_total(values):
    with np.errstate(over='ignore',invalid='ignore'):total=values.sum()
    if not np.isfinite(total)or total<=0:raise ValueError('Finite positive total quadrature mass')
    return total


def layout_probabilities(log_evidence,prior):
    """Evidence MUST include each category's normalized original parameter prior."""
    evidence=np.asarray(log_evidence,float);prior=np.asarray(prior,float)
    if evidence.ndim!=1 or prior.shape!=evidence.shape or not np.isfinite(evidence).all()or not np.isfinite(prior).all()or np.any(prior<=0)or not np.isclose(_positive_total(prior),1,rtol=0,atol=1e-14):raise ValueError('Finite normalized categorical prior and normalized evidence')
    logs=np.log(prior)+evidence;return np.exp(logs-logsumexp(logs))


@dataclass
class PredictionBank:
    probability:np.ndarray
    layout:np.ndarray
    candidate:np.ndarray
    target:np.ndarray
    def __post_init__(self):
        self.probability=np.asarray(self.probability,float);self.layout=np.asarray(self.layout);self.candidate=np.asarray(self.candidate,float);self.target=np.asarray(self.target,float)
        n=len(self.probability)
        if self.probability.shape!=(n,)or self.layout.shape!=(n,)or self.candidate.ndim!=2 or self.target.ndim!=2 or self.candidate.shape[0]!=n or self.target.shape[0]!=n or not n or not self.candidate.shape[1]or not self.target.shape[1]:raise ValueError('Positive finite-bank dimensions')
        if any(not np.isfinite(v).all()for v in(self.probability,self.candidate,self.target))or np.any(self.probability<0):raise ValueError('Finite positive quadrature measure/predictions')
        if self.layout.dtype.kind not in'iu' or np.any(self.layout<0):raise ValueError('Nonnegative categorical layout labels')
        self.probability=self.probability/_positive_total(self.probability)


def combine_layouts(log_evidence,prior,conditional_probabilities,candidates,targets):
    rho=layout_probabilities(log_evidence,prior)
    if len(conditional_probabilities)!=len(rho)or len(candidates)!=len(rho)or len(targets)!=len(rho):raise ValueError('All predeclared layouts retained')
    weights=[];labels=[]
    for k,p in enumerate(conditional_probabilities):
        p=np.asarray(p,float)
        if p.ndim!=1 or not p.size or not np.isfinite(p).all()or np.any(p<0):raise ValueError('Qualified positive conditional posterior quadrature')
        weights.append(rho[k]*(p/_positive_total(p)));labels.append(np.full(len(p),k,dtype=int))
    return PredictionBank(np.concatenate(weights),np.concatenate(labels),np.concatenate(candidates),np.concatenate(targets))


def variance_decomposition(bank):
    p,t=bank.probability,bank.target;mean=p@t;total=float(np.mean(p@((t-mean)**2)));within=0.;between=0.
    for k in np.unique(bank.layout):
        take=bank.layout==k;mass=p[take].sum()
        if mass==0:continue
        q=p[take]/mass;mu=q@t[take];within+=mass*float(np.mean(q@((t[take]-mu)**2)));between+=mass*float(np.mean((mu-mean)**2))
    return dict(total=total,within_layout=within,between_layout=between,mean=mean)


def outcome_risk(bank,action,y,noise_SD,*,latent_chunk=4096):
    """Centered two-pass posterior moments; all latent hypotheses retained."""
    y=np.asarray(y,float)
    if y.ndim!=1 or not np.isfinite(y).all()or not np.isfinite(noise_SD)or noise_SD<=0 or not isinstance(latent_chunk,int)or latent_chunk<=0 or not 0<=action<bank.candidate.shape[1]:raise ValueError('Finite scalar observations/noise and bounded latent chunks')
    p=bank.probability;x=bank.candidate[:,action];t=bank.target;offset=p@t;logp=np.full(len(p),-np.inf);np.log(p,out=logp,where=p>0);peak=np.full(len(y),-np.inf)
    def logs(start):
        stop=min(start+latent_chunk,len(p));return stop,logp[start:stop,None]-.5*((x[start:stop,None]-y)/noise_SD)**2-np.log(noise_SD*np.sqrt(2*np.pi))
    for start in range(0,len(p),latent_chunk):
        _,ll=logs(start);peak=np.maximum(peak,np.max(ll,axis=0))
    mass=np.zeros(len(y));first=np.zeros((len(y),t.shape[1]))
    for start in range(0,len(p),latent_chunk):
        stop,ll=logs(start);w=np.exp(ll-peak);mass+=w.sum(axis=0);first+=w.T@(t[start:stop]-offset)
    if not np.isfinite(peak).all()or np.any(mass<=0):raise ValueError('Unresolved predictive outcome density')
    mean=first/mass[:,None];variance=np.zeros(len(y))
    for start in range(0,len(p),latent_chunk):
        stop,ll=logs(start);w=np.exp(ll-peak)
        for j in range(len(y)):variance[j]+=np.dot(w[:,j],np.mean((t[start:stop]-offset-mean[j])**2,axis=1))/mass[j]
    return peak+np.log(mass),variance


def expected_risk(bank,action,noise_SD,*,order=16,panel_noise_width=.5,tail_sigmas=8.,outcome_chunk=16,latent_chunk=4096,maximum_outcome_nodes=200000):
    """Positive GL outcome integration plus explicit finite-bank Gaussian-tail bound.

    Outside [min(candidate)-L*sigma,max(candidate)+L*sigma], predictive mass
    is at most 2 Phi(-L). Every conditional target variance is bounded by the
    sum of squared target coordinate ranges/(4*target_dimension).
    """
    if not isinstance(order,int)or order<2 or not np.isfinite(noise_SD)or noise_SD<=0 or not np.isfinite(panel_noise_width)or panel_noise_width<=0 or not np.isfinite(tail_sigmas)or tail_sigmas<4 or not isinstance(outcome_chunk,int)or outcome_chunk<=0 or not isinstance(maximum_outcome_nodes,int)or maximum_outcome_nodes<=0:raise ValueError('Positive deterministic outcome integration plan')
    if not 0<=action<bank.candidate.shape[1]:raise ValueError('Candidate index')
    x=bank.candidate[:,action];lower=x.min()-tail_sigmas*noise_SD;upper=x.max()+tail_sigmas*noise_SD;panels=max(1,int(np.ceil((upper-lower)/(panel_noise_width*noise_SD))))
    if panels*order>maximum_outcome_nodes:raise MemoryError('Prospective scalar-outcome node budget exceeded; unresolved, no action filtering')
    nodes,weights=np.polynomial.legendre.leggauss(order);edges=np.linspace(lower,upper,panels+1);centres=(edges[:-1]+edges[1:])/2;half=np.diff(edges)/2;ys=(centres[:,None]+half[:,None]*nodes).ravel();ws=(half[:,None]*weights).ravel();risk=0.;mass=0.
    for begin in range(0,len(ys),outcome_chunk):
        end=begin+outcome_chunk;lp,v=outcome_risk(bank,action,ys[begin:end],noise_SD,latent_chunk=latent_chunk);w=ws[begin:end]*np.exp(lp);risk+=float(w@v);mass+=float(w.sum())
    bound=float(np.mean(np.ptp(bank.target,axis=0)**2)/4);tail=float(2*ndtr(-tail_sigmas));prior=variance_decomposition(bank)['total']
    if not np.isfinite(risk)or risk<0 or mass<=0 or mass>1+1e-10:raise ValueError('Unresolved outcome integral; no clipping')
    return dict(expected_target_variance=risk,prior_target_variance=prior,variance_reduction=prior-risk,predictive_mass=mass,omitted_mass_bound=tail,omitted_risk_bound=tail*bound,outcome_nodes=len(ys),order=order)


def screen_actions(bank,noise_SD,*,orders=(8,16,32),relative_error_gate=1e-4,tie_order=None):
    """Prospective finite-bank screen, not a closed-loop performance claim."""
    if len(orders)<2 or any(not isinstance(v,int)or v<2 for v in orders)or any(a>=b for a,b in zip(orders,orders[1:]))or not np.isfinite(relative_error_gate)or relative_error_gate<=0:raise ValueError('Successive positive outcome orders')
    count=bank.candidate.shape[1]
    tie_order=list(range(count))if tie_order is None else list(tie_order)
    if len(tie_order)!=count or set(tie_order)!=set(range(count)):raise ValueError('Predeclared candidate tie-order permutation')
    prior=variance_decomposition(bank)['total'];records=[];qualified=True
    for action in range(bank.candidate.shape[1]):
        levels=[expected_risk(bank,action,noise_SD,order=n)for n in orders];last=levels[-1];error=abs(last['expected_target_variance']-levels[-2]['expected_target_variance'])+last['omitted_risk_bound'];passed=error<=relative_error_gate*prior if prior>0 else error==0
        qualified=qualified and passed;records.append(dict(action=action,levels=levels,successive_error=error,qualified=passed))
    final=np.array([r['levels'][-1]['expected_target_variance']for r in records]);errors=np.array([r['successive_error']for r in records]);winner=int(np.argmin(final));ties=np.flatnonzero(final-final[winner]<=errors+errors[winner]).tolist()
    return dict(qualified=bool(qualified),records=records,minimum_risk_action=winner if qualified else None,unique_best_action=winner if qualified and len(ties)==1 else None,selected_action=next(a for a in tie_order if a in ties)if qualified else None,numerically_indistinguishable_actions=ties,total_variance=prior,scope='Finite positive prefix posterior; outcome quadrature only. Parameter/PDE admission and held-test performance are separate.')


def prospective_design():
    return dict(status='SOURCE_ONLY_MECHANICS_AND_INFERENCE_ADMISSION_REQUIRED',
        latent='Predeclared categorical support layout with uniform prior; normalized loguniform E/K prior within every layout. No category dropped.',
        prefix='Fixed partial marker channels, registered before any development results; same observation for all models and policies.',
        action_sensor='ONE force-aligned scalar marker displacement per poke, unit projection of isotropic20um marker noise. This is a smaller sensor budget than the old24channel study.',
        action_budget='Same integral of traction magnitude, patch area, reset duration and scalar sensing budget across candidates. Net force and torque reported separately; no force/energy matching claim from equal net force alone.',
        acquisition='Expected mean heldload predictive variance after one independently noisy poke, including within-layout and between-layout uncertainty. No true layout, true parameters or actual future response enters selection.',
        quadrature='Positive parameter quadrature separately qualified for every category; deterministic scalar outcome GL8/16/32 with panelwidth sigma/2 and8sigma finite-bank tail bound. Successive differences are convergence witnesses, not proofs.',
        decision_admission='Require parameter-bank and outcome refinements plus independent scalar adaptive integration controls. Unique winner requires gap exceeding combined numerical uncertainty; ties use a preregistered catalog order.',
        baselines=['Strong fixed candidate tuned on ALL development objects only','Exact uniformrandom policy expectation over all candidates','Preregistered geometry heuristic','Local Fisher approximation as explicit cheap baseline'],
        development='All deterministic layout/E/K development points retained with ranking/ties/unresolved counts and regret. Universal ranking triggers a stated scenario-redesign decision before test protocol; no test objects selected for oracle wins.',
        test='Separate fresh sealed prior-predictive cohort protocol frozen before draws; same objects/prefix/probe-noise common random controls across candidate outcomes. No adaptive advantage claim from development ranking crossover.',
        reporting='Paired object-cluster target prediction differences against each fixed policy separately; all failed objects retained. One-step probing only, no multi-step closed-loop claim.',
        resources='No actual physics/inference/draws authorized by this source. Pilot must bound per-category inverse, outcome integration, latent-bank storage and independent controls before campaign release.')
