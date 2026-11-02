import numpy as np
import pytest
from scripts.compositional_neural_mechanics_reporting.unknown_reaction_secondary import validate_completion_rows,validate_origins,primary

def test_exact_completion_census():
 rows=[dict(world=2,model='shared_laws_fitted',state_name=s,command=j,state='complete')for s in ('initial',*primary.ACTIONS)for j in (0,1)]
 validate_completion_rows(rows,2,'shared_laws_fitted')
 with pytest.raises(ValueError):validate_completion_rows(rows[:-1]+rows[:1],2,'shared_laws_fitted')
 with pytest.raises(ValueError):validate_completion_rows(rows,1,'shared_laws_fitted')

def test_origin_geometry_and_pose():
 public=dict(geometry=dict(dimensions=[.2,.08,.08],taper=0.,arch=0.),held_poses=np.zeros((2,2,6)).tolist())
 origins=np.array([[0,0,0],[.2,0,0]])
 validate_origins(origins,public,0)
 with pytest.raises(ValueError):validate_origins(origins+.01,public,0)
 with pytest.raises(ValueError):validate_origins(np.full((2,3),np.nan),public,0)
