import json
import pytest
from scripts.physical_refinement_v15.render_extra_bank import names_from_plan
from scripts.physical_refinement_v15.quadrature import geometry_cases
from scripts.physical_refinement_v15.study import base_cases
from scripts.physical_refinement_v15.render_bank import ROOT,sha

def test_only_exact_declared_new_names_allowed(tmp_path):
 old={c['name']for c in base_cases()};names=[c['name']for c in geometry_cases(9)if c['name']not in old]
 p=tmp_path/'plan.json';record=dict(case_names=names,quadrature_sha256=sha(ROOT/'scripts/physical_refinement_v15/quadrature.py'));p.write_text(json.dumps(record));assert len(names_from_plan(p))==16
 record['case_names'][0]=next(iter(old));p.write_text(json.dumps(record))
 with pytest.raises(ValueError,match='old20'):names_from_plan(p)
