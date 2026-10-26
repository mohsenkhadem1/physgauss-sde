"""Manufactured resource/identity controls; no actual noise or outcomes."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments import run_noisy_poke_continuation_v02 as run


class ContinuationTests(unittest.TestCase):
    def test_science_functions_are_original(self):
        for name in ('infer','load_common','query_truth','snapshot_metrics','observations_fingerprint'):
            self.assertIs(getattr(run,name),getattr(run.parent,name))
        delta={k:(run.parent.RESOURCES[k],v)for k,v in run.RESOURCES.items()if v!=run.parent.RESOURCES[k]}
        self.assertEqual(delta,{'evaluation_wall_s':(10800,18000),'global_wall_s':(14400,24000)})

    def test_only_trial_zero_reuses_original_directory(self):
        self.assertEqual(run.evaluation_directory(0),run.parent.LOCAL/'trial-00/evaluation')
        self.assertEqual(run.evaluation_directory(31),run.LOCAL/'trial-31/evaluation')
        self.assertNotEqual(run.LOCAL,run.parent.LOCAL)

    def cost(self,walls,qualified=True):
        trial=dict(qualified=qualified,posteriors=[dict(action=i,qualified=True,record={'path':str(i)})for i in range(8)])
        worker=dict(completed=True,trials=[dict(index=0,record={'path':'trial'})])
        def read(path):
            return worker if str(path)=='worker'else trial if str(path)=='trial'else {'wall_s':walls[int(str(path))]}
        with patch.object(run.parent,'completed_phase',return_value={'workers':[{'result':{'path':'worker'}}]}),patch.object(run,'exact',side_effect=lambda r:r['path']),patch.object(run,'read',side_effect=read):
            return run.checkpoint_cost('manufactured')

    def test_measured_original_checkpoint_fits_new_cap(self):
        result=self.cost([49.320520]*8)
        self.assertAlmostEqual(result['projected_remaining_wall_s'],13020.61728)
        self.assertGreater(result['projected_remaining_wall_s'],result['original_budget_s'])
        self.assertLess(result['projected_remaining_wall_s'],result['revised_budget_s'])

    def test_cost_still_rejects_over_budget_and_nonfinite(self):
        for value in (100.,float('nan'),-1.):
            with self.subTest(value=value),self.assertRaises(ValueError):self.cost([value]*8)

    def test_unqualified_checkpoint_cannot_continue(self):
        with self.assertRaises(ValueError):self.cost([1.]*8,False)

    def test_no_checkpoint_or_choice_worker_rerun(self):
        for phase in ('choices','checkpoint'):
            with self.subTest(phase=phase),self.assertRaises(ValueError):run.evaluate_worker(0,phase,'manufactured')

    def test_refuse_prior_remaining_attempt_or_partial_trial(self):
        with tempfile.TemporaryDirectory()as folder,patch.object(run.parent,'LOCAL',Path(folder)):
            run.untouched_original_remaining()
            path=Path(folder)/'execution-remaining.json';path.write_text('{}')
            with self.assertRaises(FileExistsError):run.untouched_original_remaining()
            path.unlink();path=Path(folder)/'trial-17/evaluation/partial.json';path.parent.mkdir(parents=True);path.write_text('{}')
            with self.assertRaises(FileExistsError):run.untouched_original_remaining()

    def test_exact_commitment_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'x';path.write_text('before')
            with patch.object(run,'ROOT',Path(folder)):
                # record() is the frozen helper, so manufacture an intentionally bad digest.
                with self.assertRaises(ValueError):run.exact({'path':'x','sha256':'0'*64})


class ExternalWatchdogTests(unittest.TestCase):
    def witness(self,change=None):
        root=Path('/manufactured-recovery');local=root/'continuous';op=root/'runs/operator-dev/noisy-poke-recovery-operator-v02'
        rec=lambda p:dict(path=str(Path(p).relative_to(root)),sha256='fixed')
        pins={'science.py':'held','experiments/supervise_noisy_poke_recovery_v02.py':'held','tests/test_supervise_noisy_poke_recovery_v02.py':'held'}
        approval=dict(approved=True,schema='noisy-poke-recovery-operator-preapproval-v02',source_SHA256=pins,stage_caps={'remaining':[18,18060,run.RESOURCES['aggregate_RSS_bytes']]})
        stage=dict(completed=True,reaped=True,exit_code=0,cleanup_exit_code=0,phase='remaining',command=['taskset','-c','18',run.sys.executable,'-u','-m','experiments.run_noisy_poke_continuation_v02','run','--sha','parent'],parent_protocol_SHA256='parent',source_SHA256=pins,operator_preapproval=rec(op/'root-preapproval.json'),resources=dict(CPU=18,wall_s=18060,RSS_bytes=run.RESOURCES['aggregate_RSS_bytes']),outputs=[rec(local/'execution-remaining.json')],child_identity={'pid':987651},wall_s=1.,peak_RSS_bytes=1,observed_owned=[{'pid':p}for p in [987651,987652]])
        if change:stage.update(change)
        closure=dict(operator_preapproval=rec(op/'root-preapproval.json'),operator_stage_execution=rec(op/'stages/remaining/execution.json'),owned_pids=[987651,987652])
        with patch.object(run,'ROOT',root),patch.object(run,'LOCAL',local),patch.object(run,'record',side_effect=rec),patch.object(run,'sources',return_value={'science.py':'held'}),patch.object(run,'sha',return_value='held'),patch.object(run,'exact',side_effect=lambda r:root/r['path']),patch.object(run,'read',side_effect=lambda p:approval if p.name=='root-preapproval.json'else stage),patch.object(run.base,'absent')as absent:
            run.external_watchdog({'controller_pid':987651},closure,'parent')
            self.assertEqual(absent.call_count,2)

    def test_exact_external_watchdog_passes(self):self.witness()

    def test_unreaped_wrong_command_unobserved_or_over_cap_rejected(self):
        for change in ({'reaped':False},{'exit_code':1},{'command':['other']},{'wall_s':18061},{'peak_RSS_bytes':run.RESOURCES['aggregate_RSS_bytes']},{'observed_owned':[{'pid':987651}]}):
            with self.subTest(change=change),self.assertRaises(ValueError):self.witness(change)

if __name__=='__main__':unittest.main()
