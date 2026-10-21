import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.campaign_numerical_convergence import stage_case,command_clock_audit,digest
from experiments.campaign_numerical_validate_rates import adapt_reset_reference


class ConvergenceTests(unittest.TestCase):
    def test_menu_adapter_retains_position_and_velocity_reset_rejection(self):
        source="def check(saved):\n a=abs(saved['poke_0']['q']-saved['reset_replay']['q'])\n b=abs(saved['poke_0']['v']-saved['reset_replay']['v'])\n if a>1e-6 or b>1e-7:raise ValueError('Bad replay')\n return True\n"
        namespace={};exec(adapt_reset_reference(source,'zero_force'),namespace)
        check=namespace['check'];self.assertTrue(check({'zero_force':{'q':1,'v':2},'reset_replay':{'q':1,'v':2}}))
        with self.assertRaises(ValueError):check({'zero_force':{'q':1,'v':2},'reset_replay':{'q':1,'v':2.01}})
        with self.assertRaises(ValueError):adapt_reset_reference(source.replace("saved['poke_0']","saved['other']",1),'zero_force')

    def test_staging_changes_only_rate_and_recipe_hash_preserving_original(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);ep=root/'dataset/episodes/t0001';(ep/'truth').mkdir(parents=True);(ep/'observations').mkdir()
            case={'split':'train','boundary_type':'fixed_patch','physics_hz':960,'seed':123,'recipe_sha256':'old'}
            (ep/'truth/recipe.json').write_text(json.dumps(case));(ep/'truth/boundaries.json').write_text('{"hardware":123}')
            (ep/'observations/cameras.npz').write_bytes(b'known-camera')
            before={p.relative_to(ep):p.read_bytes() for p in ep.rglob('*') if p.is_file()}
            staged=stage_case(root/'dataset',root/'audit','t0001',480)/'episodes/t0001'
            modified=json.loads((staged/'truth/recipe.json').read_text());h=modified.pop('recipe_sha256');self.assertEqual(h,digest(modified))
            modified['physics_hz']=960;modified['recipe_sha256']='old';self.assertEqual(modified,case)
            self.assertEqual(before,{p.relative_to(ep):p.read_bytes() for p in ep.rglob('*') if p.is_file()})
            self.assertEqual((staged/'observations/cameras.npz').read_bytes(),before[Path('observations/cameras.npz')])

    def test_midpoint_impulse_and_float32_clock_at_each_rate(self):
        for hz in [480,960,1920]:
            dt32=float(np.float32(1/hz));mid=(np.arange(3*hz)+.5)/hz;phase=(mid/3-.1)/.45
            envelope=np.where((phase>0)&(phase<1),np.sin(np.pi*np.clip(phase,0,1))**2,0)
            force=np.zeros((len(mid),2,3),np.float32);peak=np.array([.01,.01,.005]);force[:,0]=envelope[:,None]*peak
            times=np.arange(31)/10;data={'times_s':times,'simulated_times_s':np.rint(times*hz)*dt32,
                'force_step_midpoints_s':mid,'force_step_commands_N':force}
            r=command_clock_audit(data,hz,{'world_get_physics_dt_s':dt32},peak)
            self.assertLess(r['nominal_impulse_max_error_Ns'],1e-9)
            data['simulated_times_s']=data['simulated_times_s']+1e-5
            with self.assertRaises(ValueError):command_clock_audit(data,hz,{'world_get_physics_dt_s':dt32},peak)


if __name__=='__main__':unittest.main()
