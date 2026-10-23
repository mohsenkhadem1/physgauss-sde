"""Focused process-race/orchestration controls; no new scientific inference."""
import ast,copy,errno,hashlib,json,sys
from pathlib import Path
import pytest
from experiments import campaign_owned_proc as proc
from experiments import run_campaign_surrogate_sbc_v02 as old
from experiments import run_campaign_surrogate_sbc_v03 as fixed
from experiments import run_campaign_surrogate_sbc_evaluation_continuation as c
from experiments import supervise_surrogate_sbc_continuation as supervisor


def test_process_exit_errors_only_ENOENT_ESRCH_are_benign(monkeypatch):
 for err in (FileNotFoundError(errno.ENOENT,'gone'),ProcessLookupError(errno.ESRCH,'exited')):
  monkeypatch.setattr(Path,'read_text',lambda*a,_error=err,**k:(_ for _ in()).throw(_error))
  assert proc.proc_identity(99999999)is None and proc.proc_rss(99999999)==0
 for err in (PermissionError(errno.EACCES,'denied'),OSError(errno.EIO,'I/O error')):
  monkeypatch.setattr(Path,'read_text',lambda*a,_error=err,**k:(_ for _ in()).throw(_error))
  with pytest.raises(OSError)as e:proc.proc_rss(1)
  assert e.value.errno==err.errno
  with pytest.raises(OSError):proc.proc_identity(1)


def test_real_child_exit_with_injected_ESRCH_in_controller_AND_supervisor(tmp_path,monkeypatch):
 original=Path.read_text;seen=[]
 def racing(path,*a,**k):
  if str(path).startswith('/proc/')and path.name=='status':seen.append(str(path));raise ProcessLookupError(errno.ESRCH,'focused normal-exit race')
  return original(path,*a,**k)
 monkeypatch.setattr(Path,'read_text',racing)
 out=tmp_path/'owned';out.mkdir();rows=fixed.owned_jobs(lambda i,s:[sys.executable,'-c','import time;time.sleep(.1)'],[0],1,out,worker_wall_s=5,global_wall_s=8,RSS_bytes=2*1024**3)
 assert rows[0]['exit_code']==0 and rows[0]['reaped']and not Path('/proc',str(rows[0]['pid'])).exists()
 result=supervisor.supervise([sys.executable,'-c','import time;time.sleep(.15)'],tmp_path/'supervisor.log',tmp_path/'receipt.json',wall_s=5)
 assert result['completed']and result['exit_code']==0 and result['controller_reaped']and result['controller_PID_absent'];assert len(seen)>=2


def test_scientific_AST_and_all_original_source_pins_unchanged():
 frozen=json.loads(Path('runs/operator-dev/matching-prior-surrogate-sbc-v02/protocol.json').read_text())
 assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in frozen['source_sha256'].items())
 assert c.old.DESIGN==frozen['design']and c.RESOURCES==frozen['resources']
 def function(module,name):return next(n for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
 a=function(old,'worker');b=function(c,'worker')
 # Evaluate branch contains reconstruction, selective truth and exact same PIT.
 original=next(n for n in a.body if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)and ast.unparse(n.test)=="stage == 'inference'").orelse
 begin=next(i for i,n in enumerate(b.body)if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='previous');continued=b.body[begin:-1]
 class StripAlias(ast.NodeTransformer):
  def visit_Attribute(self,n):
   n=self.generic_visit(n)
   if isinstance(n.value,ast.Name)and n.value.id=='old':return ast.copy_location(ast.Name(id=n.attr,ctx=n.ctx),n)
   return n
 clean=lambda seq:ast.dump(StripAlias().visit(ast.Module(body=copy.deepcopy(seq),type_ignores=[])),include_attributes=False)
 assert clean(original)==clean(continued)
 # Pure scientific scorer is imported unchanged, never cloned or edited.
 assert c.sbc is old.sbc


def test_all64_merge_retains_complete_score_identity_and_negative_denominators(tmp_path):
 def value(i):
  score=dict(completed=False,error='Manufactured retained negative')
  e=dict(methods={m:copy.deepcopy(score)for m in c.sbc.METHODS},numerical_controls_passed=False,maximum_method_PIT_difference=None)
  return dict(object_index=i,phases={p:dict(object_index=i,admitted=False,evaluation=copy.deepcopy(e))for p in c.sbc.PHASES})
 rows=[];original={}
 for i in range(64):
  v=value(i);original[i]=v;files=old.bundle(tmp_path,tmp_path/'bundles'/str(i),v);rows.append(dict(object_index=i,files=files,origin='REUSED_COMPLETE_V02'if i<9 else'CONTINUED_SAME_COHORT'))
 p=dict(inputs=dict(reused=rows[:9],remaining_indices=list(range(9,64))));new=[dict(object_index=r['object_index'],files=r['files'],exit_code=0,reaped=True)for r in rows[9:]];merged,values=c.merged_rows(tmp_path,p,new)
 assert len(merged)==64 and all(old.exact_tree_identity(v,original[v['object_index']])for v in values)
 for phase in c.sbc.PHASES:
  a=c.sbc.aggregate([v['phases'][phase]for v in values]);assert a['objects']==64 and a['numerically_admitted']==0 and all(q['failed']==64 for q in a['quantities'].values())
 with pytest.raises(ValueError):c.merged_rows(tmp_path,p,new[:-1])
