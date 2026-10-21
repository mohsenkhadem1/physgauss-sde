import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
from freeze_gaussian_final_catalog import freeze


class FinalCatalogTests(unittest.TestCase):
    def dataset(self,root,overlap=False):
        episodes=[{'episode_id':'t0','split':'train','group_id':'a','recipe_sha256':'t'},
                  {'episode_id':'v0','split':'validation','group_id':'a' if overlap else 'b','recipe_sha256':'v'},
                  {'episode_id':'z0','split':'test'}]
        (root/'plan.json').write_text(json.dumps({'dataset':'unit','episodes':episodes}))
        for episode in ('t0','v0'):
            folder=root/'episodes'/episode;folder.mkdir(parents=True)
            (folder/'summary.json').write_text('{"completed":true}')
            (folder/'validation.json').write_text('{"passed":true}')

    def test_requires_complete_supervision_and_never_requires_test_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.dataset(root);output=root/'catalog.json'
            (root/'episodes/v0/failure.json').write_text('{}')
            with self.assertRaises(ValueError):freeze([root],output,1,1)
            self.assertFalse(output.exists())
            (root/'episodes/v0/failure.json').unlink()
            result=freeze([root],output,1,1)
            self.assertEqual(len(result['cases']),2);self.assertFalse((root/'episodes/z0').exists())
            with self.assertRaises(FileExistsError):freeze([root],output,1,1)

    def test_rejects_group_overlap_and_wrong_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.dataset(root,overlap=True)
            with self.assertRaises(ValueError):freeze([root],root/'catalog.json',1,1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.dataset(root)
            with self.assertRaises(ValueError):freeze([root],root/'catalog.json',2,1)


if __name__=='__main__':unittest.main()
