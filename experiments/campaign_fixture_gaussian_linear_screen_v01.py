"""NEW exact full-mode adapter for Gaussian-v02 fixture DEVELOPMENT screen.

Versioned resolution queries are explicit; no frozen module globals changed.
No mechanics executes on import. Geometry/support/loads and all modes intact.
"""
from types import SimpleNamespace
import numpy as np
from experiments import campaign_active_fixture_resolution_v03 as geometry
from experiments import campaign_active_fixture_gaussian_load_v02 as bank
from experiments.campaign_surface_p2_inference_v01 import LabelledModal


def make_measurement_modal(forward,loads):
    points=bank.measurement_points()
    base=LabelledModal.from_forward(forward,loads=loads,points=points,condensed=True,query_matrix=geometry.point_query_matrix)
    specs=bank.measurement_specs();rows=[]
    for spec in specs:
        point=next(i for i,p in enumerate(points) if np.array_equal(p,spec['point_m']));load=base.labels.index(spec['load'])
        rows.append(np.asarray(spec['projection'])@base.coefficients[3*point:3*point+3,load,:])
    modal=SimpleNamespace(eigenvalues=base.modal.eigenvalues.copy(),coefficients=np.asarray(rows),output_shape=(1,len(specs)),diagnostics=dict(base.modal.diagnostics,measurement_selection='All60public point/projection rows; versioned arbitrary-declared-n query; all modes and exact Schur1/E particular retained'))
    return LabelledModal(modal,tuple(s['id'] for s in specs))


def measurement_response(n,fields):
    columns=[]
    for spec in bank.measurement_specs():
        u=np.asarray(fields[spec['load']],float)
        if u.ndim!=3:raise ValueError('Explicit [batch,Nnodes,3]')
        value=geometry.field_queries(n,u,np.array([spec['point_m']]))[:,0]@np.asarray(spec['projection'])
        columns.append(value)
    return np.stack(columns,axis=1)[:,:,None]


def worklist():
    center=(4000.,float(np.sqrt(150.)))
    grid=[(E,K) for E in (2000.,4000.,8000.) for K in (5.,float(np.sqrt(150.)),30.)]
    parameters=[center]+[p for p in grid if p!=center]
    return [dict(index=i,layout=s['id'],E_Pa=E,K_N_m=K,n=n) for i,(s,E,K,n) in enumerate((s,E,K,n) for s in bank.LAYOUTS for E,K in parameters for n in (3,6))]


def separation(rows):
    answer=[]
    for n in (3,6):
        for E in (2000.,4000.,8000.):
            for K in (5.,float(np.sqrt(150.)),30.):
                group=[r for r in rows if (r['n'],r['E_Pa'],r['K_N_m'])==(n,E,K)]
                if len(group)!=4:raise ValueError('All four categories required; never drop a missing layout')
                for j,a in enumerate(group):
                    for b in group[j+1:]:
                        d=np.asarray(a['response'])-np.asarray(b['response'])
                        answer.append(dict(n=n,E_Pa=E,K_N_m=K,layouts=[a['layout'],b['layout']],prefix4_RMS_m=float(np.sqrt(np.mean(d[:4]**2))),prefix4_L2_over_sigma=float(np.linalg.norm(d[:4])/bank.SIGMA_M),candidate8_RMS_m=float(np.sqrt(np.mean(d[4:12]**2))),held48_RMS_m=float(np.sqrt(np.mean(d[12:]**2))),held48_RMS_over_sigma=float(np.sqrt(np.mean(d[12:]**2))/bank.SIGMA_M)))
    return answer
