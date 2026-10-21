import sys,json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
from evaluate_gaussian_inverse import load_public_observations,encode_inputs
from train_gaussian_inverse import snapshot_catalog,apply_catalog,collect_cases
from isaac_dataset import make_plan,geometry,camera_pose,appearance


class EvaluationIsolationTests(unittest.TestCase):
    def test_prefix_never_opens_future_images(self):
        p,t,f,spec=geometry(make_plan()['episodes'][0])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'observations/rgb').mkdir(parents=True);(root/'observations/masks').mkdir()
            np.savez(root/'reference.npz',rest_points_m=p,surface_faces=f)
            np.savez(root/'observations/cameras.npz',frame_ids=[0,40],times_s=[0.,2.],
                     K=np.array([[622.,0,320],[0,622.,240],[0,0,1.]]),camera_to_world_usd_row=np.array([camera_pose(spec,0),camera_pose(spec,2)]))
            Image.fromarray(np.full((480,640,3),200,np.uint8)).save(root/'observations/rgb/00000.png')
            Image.fromarray(np.full((480,640),255,np.uint8)).save(root/'observations/masks/00000.png')
            result=load_public_observations(root,max_time_s=1.)
            self.assertEqual(result['cameras']['frame_ids'].tolist(),[0])
            self.assertEqual(result['rgb_native'].shape,(1,480,640,3))
            self.assertEqual(result['native_mask'].shape,(1,480,640))
            self.assertFalse((root/'truth').exists())
            with self.assertRaises(FileNotFoundError):load_public_observations(root)

    def test_texture_checkpoint_appends_features_from_public_arrays(self):
        import torch
        from gaussian_texture_features import fit_color_curve
        p,t,f,spec=geometry(make_plan()['episodes'][0]);colors,g=appearance(p,f)
        w=np.linspace(0,1,20);curve=fit_color_curve(w,np.stack((.3+.5*w,.5+.2*w,.8-.4*w),-1))
        class Capture:
            def context(self,batch):return batch['features'].mean(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'observations/rgb').mkdir(parents=True);(root/'observations/masks').mkdir()
            np.savez(root/'reference.npz',rest_points_m=p,surface_faces=f,vertex_colors=colors,**g)
            np.savez(root/'observations/cameras.npz',frame_ids=[0],times_s=[0.],
                     K=np.array([[622.,0,320],[0,622.,240],[0,0,1.]]),camera_to_world_usd_row=np.array([camera_pose(spec,0)]))
            Image.fromarray(np.full((480,640,3),200,np.uint8)).save(root/'observations/rgb/00000.png')
            Image.fromarray(np.full((480,640),255,np.uint8)).save(root/'observations/masks/00000.png')
            inputs=load_public_observations(root)
            bundle={'device':torch.device('cpu'),'inverse':Capture(),'checkpoint':{
                'model_kwargs':{'input_mode':'features','observation_mode':'rgb'},
                'texture_calibration':{'global_curve':curve,'face_curves':{}}}}
            encoded=encode_inputs(bundle,inputs)
            self.assertEqual(encoded['context'].shape,(1,143))
            self.assertTrue(torch.isfinite(encoded['context']).all())
            inputs.pop('rgb_native')
            with self.assertRaises(ValueError):encode_inputs(bundle,inputs)

    def test_catalog_is_immutable_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'episodes/t0000').mkdir(parents=True)
            (root/'episodes/t0000/summary.json').write_text('{"completed":true}')
            (root/'episodes/t0000/validation.json').write_text('{"passed":true}')
            (root/'plan.json').write_text(json.dumps({'dataset':'unit','episodes':[
                {'episode_id':'t0000','split':'train','group_id':'g0','recipe_sha256':'abc'},
                {'episode_id':'z0000','split':'test'}]}))
            path=root/'catalog.json';snapshot_catalog([root],path)
            with self.assertRaises(FileExistsError):snapshot_catalog([root],path)
            self.assertEqual(len(apply_catalog(collect_cases([root]),path)),1)
            raw=json.loads(path.read_text());raw['cases'][0]['recipe_sha256']='changed';path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):apply_catalog(collect_cases([root]),path)

if __name__=='__main__':unittest.main()
