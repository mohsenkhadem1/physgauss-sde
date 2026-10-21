import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import numpy as np
from experiments.repair_isaac_dataset_sensors import (prepare_repair,tree_hashes,refresh_sparse_copies,load_arrays,rest_projection_mask,audit_frame,repair_batch)

class SensorRepairTests(unittest.TestCase):
    def test_original_preserved_exactly_and_working_copy_is_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);episode=root/'episodes/t0053';episode.mkdir(parents=True)
            (root/'logs').mkdir();(root/'logs/t0053.status.json').write_text(json.dumps({'status':'failed'}))
            (root/'logs/t0053.log').write_text('original rejected run\n')
            (episode/'summary.json').write_text('{"completed":true}')
            (episode/'pixels.bin').write_bytes(b'original pixels')
            expected=tree_hashes(episode)
            rejected,working,metadata,recorded=prepare_repair(root,'t0053')
            self.assertFalse(episode.exists());self.assertEqual(tree_hashes(rejected),expected)
            self.assertEqual(tree_hashes(working),expected);self.assertEqual(recorded,expected)
            (working/'pixels.bin').write_bytes(b'replacement')
            self.assertEqual(tree_hashes(rejected),expected)
            self.assertEqual(json.loads((metadata/'original-file-sha256.json').read_text()),expected)
            self.assertEqual((metadata/'original-worker.log').read_text(),'original rejected run\n')
            self.assertEqual(json.loads((root/'logs/t0053.status.json').read_text())['status'],'sensor_repair_pending')

    def test_failed_batch_capture_preserves_all_originals_without_promotion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'dataset';(root/'logs').mkdir(parents=True);rows=[];expected={}
            for name in ['t0054','t0057']:
                episode=root/'episodes'/name;episode.mkdir(parents=True)
                (episode/'summary.json').write_text('{"completed":true}')
                (episode/'pixels.bin').write_bytes(name.encode())
                (root/'logs'/(name+'.status.json')).write_text('{"status":"generated_verified"}')
                expected[name]=tree_hashes(episode);rows.append({'dataset':str(root),'episode_id':name})
            spec=Path(temporary)/'batch.json';spec.write_text(json.dumps(rows))
            with patch('experiments.repair_isaac_dataset_sensors.audit_episode',return_value={'bad_frames':[9]}),patch('experiments.repair_isaac_dataset_sensors.subprocess.run',side_effect=subprocess.CalledProcessError(1,['renderer'])):
                with self.assertRaises(subprocess.CalledProcessError):repair_batch(spec,Path(temporary)/'out')
            for name in expected:
                self.assertEqual(tree_hashes(root/'rejected-attempts/stale-sensor'/name),expected[name])
                self.assertEqual(tree_hashes(root/'sensor-repair-working'/name),expected[name])
                self.assertFalse((root/'episodes'/name).exists())
                self.assertEqual(json.loads((root/'logs'/(name+'.status.json')).read_text())['status'],'sensor_repair_pending')

    def test_only_repaired_observed_frames_refresh_public_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for domain in ['truth','observations']:
                for folder in ['rgb','masks','depth']:(root/domain/folder).mkdir(parents=True)
            np.savez(root/'observations/cameras.npz',frame_ids=np.array([0,2]))
            for frame in [0,1,2]:
                for folder,extension in [('rgb','.png'),('masks','.png'),('depth','.npz')]:
                    (root/'truth'/folder/(f'{frame:05d}'+extension)).write_bytes(f'truth{frame}'.encode())
                    if frame in [0,2]:(root/'observations'/folder/(f'{frame:05d}'+extension)).write_bytes(b'old')
            camera=(root/'observations/cameras.npz').read_bytes()
            refreshed=refresh_sparse_copies(root,[1,2])
            self.assertEqual(len(refreshed),3)
            self.assertEqual((root/'observations/rgb/00000.png').read_bytes(),b'old')
            self.assertEqual((root/'observations/rgb/00002.png').read_bytes(),b'truth2')
            self.assertFalse((root/'observations/rgb/00001.png').exists())
            self.assertEqual((root/'observations/cameras.npz').read_bytes(),camera)

    def test_camera_clock_alignment_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'truth').mkdir()
            np.savez(root/'reference.npz',rest_points_m=np.zeros((4,3)))
            np.savez(root/'truth/trajectory.npz',times_s=[0.,1.],positions_m=np.zeros((2,4,3)))
            np.savez(root/'truth/cameras.npz',times_s=[0.,1.],frame_ids=[0,1])
            load_arrays(root)
            np.savez(root/'truth/cameras.npz',times_s=[0.,.9],frame_ids=[0,1])
            with self.assertRaises(AssertionError):load_arrays(root)

    def test_independent_rays_reject_previous_state_depth(self):
        points=np.array([[-.1,-.1,0.],[.1,-.1,0.],[.1,.1,0.],[-.1,.1,0.]])
        ref={'rest_points_m':points,'surface_faces':np.array([[0,1,2],[0,2,3]])}
        pose=np.eye(4);pose[3,2]=1.;k=np.array([[600.,0,320],[0,600.,240],[0,0,1.]])
        mask=rest_projection_mask(points,ref['surface_faces'],pose,k)
        depth=mask.astype(np.float32);rgb=np.zeros((480,640,3),np.uint8)
        self.assertTrue(audit_frame(rgb,mask,depth,points,ref,pose,k)['passed'])
        shifted=points+np.array([0.,0.,.003])
        stale=audit_frame(rgb,mask,depth,shifted,ref,pose,k)
        self.assertFalse(stale['passed']);self.assertGreater(stale['regions'][0]['maximum_error_m'],.0029)
        fresh=depth*.997
        self.assertTrue(audit_frame(rgb,mask,fresh,shifted,ref,pose,k)['passed'])
        # Large valid motion can leave all rest-mask rays on background.
        # Geometry-predicted background is checked, not treated as bad depth.
        translated=points+np.array([.3,0.,0.])
        moved_mask=rest_projection_mask(translated,ref['surface_faces'],pose,k)
        self.assertTrue(audit_frame(rgb,moved_mask,moved_mask.astype(np.float32),translated,ref,pose,k)['passed'])

if __name__=='__main__':unittest.main()
