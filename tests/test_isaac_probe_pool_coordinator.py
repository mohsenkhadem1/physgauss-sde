from pathlib import Path
import sys
import tempfile
import threading
import time
import signal
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from isaac_probe_pool_coordinator import choose_cases,run_fixed_queue,archive_interrupted_case,takeover_driver

class ProbeCoordinatorTests(unittest.TestCase):
    def test_fixed_selection_no_implicit_test_or_split_mixing(self):
        plan={'episodes':[{'episode_id':'a','split':'train'},{'episode_id':'b','split':'test'}]}
        self.assertEqual(choose_cases(plan,'train')[0]['episode_id'],'a')
        self.assertEqual(choose_cases(plan,'test',['b'])[0]['episode_id'],'b')
        for split,ids in [('test',None),('train',['b']),('test',['b','b'])]:
            with self.assertRaises(ValueError):choose_cases(plan,split,ids)

    def test_each_case_once_one_active_worker_per_gpu_failure_preserved(self):
        mutex=threading.Lock();active={0:0,2:0,3:0};maximum={k:0 for k in active};seen=[]
        def invoke(case,gpu):
            with mutex:active[gpu]+=1;maximum[gpu]=max(maximum[gpu],active[gpu]);seen.append(case['episode_id'])
            time.sleep(.003)
            with mutex:active[gpu]-=1
            if case['episode_id']=='e3':raise ValueError('Preserved physics failure')
            return {'episode_id':case['episode_id'],'gpu':gpu,'passed':True}
        results=run_fixed_queue([{'episode_id':f'e{i}'} for i in range(16)],[0,2,3],invoke)
        self.assertEqual(len(set(seen)),16);self.assertEqual(maximum,{0:1,2:1,3:1})
        self.assertEqual(sum(not r['passed'] for r in results),1)

    def test_stop_finishes_active_cases_without_new_work(self):
        stop=threading.Event()
        def invoke(case,gpu):stop.set();return {'episode_id':case['episode_id'],'passed':True}
        results=run_fixed_queue([{'episode_id':str(i)} for i in range(10)],[0],invoke,stop)
        self.assertEqual(len(results),1)

    def test_interrupted_artifacts_preserved_without_modification(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for rel in ['private/a/reference.npz','outcomes/a/frame.png','public/a/menu.json','logs/a.log']:
                p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'original bytes')
            archive=Path(archive_interrupted_case(root,'a'))
            self.assertEqual((archive/'private/a/reference.npz').read_bytes(),b'original bytes')
            self.assertFalse((root/'private/a').exists());self.assertTrue((archive/'reason.json').exists())

    def test_takeover_signals_only_driver_after_child_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);events=[];terminated=False
            def info(pid):
                nonlocal terminated
                if pid==100:return {'pid':100,'state':'Z' if terminated else 'S','command':['python','isaac_calibrated_probes_v02.py','--output',td]}
                return {'pid':200,'state':'Z','command':['isaac','isaac_calibrated_probes_v02.py','--episode','a']}
            def kill(pid,sig):
                nonlocal terminated
                events.append((pid,sig))
                if sig==signal.SIGTERM:terminated=True
            with patch('isaac_probe_pool_coordinator.process_info',side_effect=info),patch('isaac_probe_pool_coordinator.os.kill',side_effect=kill),patch.object(Path,'read_text',return_value='200'):
                ledger=takeover_driver(100,root)
            self.assertEqual(events,[(100,signal.SIGSTOP),(100,signal.SIGTERM),(100,signal.SIGCONT)])
            self.assertEqual(ledger['completed_child_episodes'],['a'])

    def test_non_driver_pid_rejected_before_signal(self):
        with patch('isaac_probe_pool_coordinator.process_info',return_value={'state':'S','command':['sleep','10']}),patch('isaac_probe_pool_coordinator.os.kill') as kill:
            with self.assertRaises(ValueError):takeover_driver(100,'/tmp')
            kill.assert_not_called()

if __name__=='__main__':unittest.main()
