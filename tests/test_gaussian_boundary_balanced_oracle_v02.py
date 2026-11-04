"""Exercise the exact frozen runner's gradient block without a FEM/GPU import."""
import ast
from pathlib import Path
import numpy as np
import pytest


def block_fixture(budget=12):
    source=Path('scripts/gaussian_boundary_flow_v02/balanced_push_oracle.py').read_text()
    function=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='block')
    code=compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'frozen-gradient-block','exec')
    class BudgetStop(Exception):pass
    points=[];calls=[];coeff=np.arange(1.,6.)
    def objective(z):
        if len(calls)>=budget:raise AssertionError('Overran hard budget')
        points.append(z.copy());calls.append({'completed':True})
        return float(np.sum(coeff*z*z))
    scope=dict(np=np,cache={},budget=budget,calls=calls,gradients=[],objective=objective,BudgetStop=BudgetStop)
    exec(code,scope)
    return scope,points,coeff


def test_exact_five_coordinate_coverage():
    scope,points,_=block_fixture();scope['block'](np.zeros(5))
    assert len(points)==6
    assert np.array_equal(np.stack(points[1:]),.001*np.eye(5))
    assert scope['gradients'][0]['coordinate_ids']==list(range(5))
    assert scope['gradients'][0]['perturbation_calls']==[2,3,4,5,6]


def test_known_quadratic_forward_difference_accuracy():
    scope,_,coeff=block_fixture();x=np.array([.2,-.5,.7,1.1,-1.5])
    loss,gradient=scope['block'](x)
    assert loss==pytest.approx(np.sum(coeff*x*x))
    assert np.allclose(gradient,2*coeff*x+.001*coeff,atol=1e-10,rtol=1e-10)


def test_budget_reservation_and_cache_never_partial_gradient():
    scope,points,_=block_fixture(budget=11)
    scope['block'](np.zeros(5));scope['block'](np.zeros(5))
    assert len(points)==6
    with pytest.raises(scope['BudgetStop']):scope['block'](np.ones(5))
    assert len(points)==6 and len(scope['gradients'])==1


def test_upper_boundary_uses_declared_negative_step():
    scope,points,coeff=block_fixture();x=np.full(5,6.)
    _,gradient=scope['block'](x)
    assert all(np.max(point)<=6. for point in points)
    assert scope['gradients'][0]['steps']==[-.001]*5
    assert np.allclose(gradient,2*coeff*x-.001*coeff,atol=1e-8)
