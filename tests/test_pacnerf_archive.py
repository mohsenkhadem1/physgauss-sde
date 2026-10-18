"""Public archives require matched provenance and never replace raw assets."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:import archive_pacnerf_checkpoints as archives
finally:sys.path.pop(0)


class ArchiveTests(unittest.TestCase):
    def prepared(self,root):
        root=Path(root);raw=root/'raw';inputs=root/'inputs'
        raw.mkdir();inputs.mkdir();payload=b'inert archive fixture'
        path=root/'pacnerf-inputs-123.zip';path.write_bytes(payload)
        digest=hashlib.sha256(payload).hexdigest()
        (inputs/'artifact_download.json').write_text(json.dumps({
            'repository':archives.REPO,'workflow_run_id':archives.STUDIES['exact']['run'],
            'artifact_name':'pacnerf-inputs','artifact_id':123,
            'archive_sha256':digest,'archive_size_bytes':len(payload)}),encoding='utf-8')
        return raw,inputs,path,digest

    def test_plan_never_authenticates_or_publishes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            raw,inputs,_,_=self.prepared(folder)
            with patch.object(archives,'github_token',side_effect=AssertionError('Plan must stay offline')):
                archives.archive(raw,inputs)
            plan=json.loads((raw/'checkpoint-archive-plan.json').read_text())
            self.assertEqual(plan['tag'],archives.STUDIES['exact']['tag'])
            self.assertEqual(len(plan['assets']),1)

    def test_other_study_provenance_rejected_before_authentication(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            raw,inputs,_,_=self.prepared(folder)
            with patch.object(archives,'github_token',side_effect=AssertionError('No authentication')):
                with self.assertRaises(ValueError):archives.archive(raw,inputs,False,'historical')

    def test_changed_zip_rejected_before_authentication(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            raw,inputs,path,_=self.prepared(folder);path.write_bytes(b'changed bytes')
            with patch.object(archives,'github_token',side_effect=AssertionError('No authentication')):
                with self.assertRaises(ValueError):archives.archive(raw,inputs,False)

    def test_existing_conflicting_asset_is_never_replaced(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            raw,inputs,path,_=self.prepared(folder);methods=[]
            release={'target_commitish':archives.STUDIES['exact']['commit'],'prerelease':True,
                'draft':False,'assets_url':'https://api.github.com/assets',
                'upload_url':'https://uploads.github.com/assets{?name}','html_url':'https://github.com/test'}
            class Opener:
                def open(self,request,timeout):
                    methods.append(request.get_method())
                    value=([{'name':path.name,'digest':'sha256:wrong','size':path.stat().st_size,'state':'uploaded'}]
                        if request.full_url.startswith(release['assets_url']) else release)
                    return io.BytesIO(json.dumps(value).encode())
            with patch.object(archives,'github_token',return_value='inert-test-placeholder'),patch.object(archives.urllib.request,'build_opener',return_value=Opener()):
                with self.assertRaises(ValueError):archives.archive(raw,inputs,False)
            self.assertEqual(methods,['GET','GET'])

    def test_initialization_only_uploads_no_assets(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            raw,inputs,_,_=self.prepared(folder);methods=[]
            class Opener:
                def open(self,request,timeout):
                    methods.append(request.get_method())
                    return io.BytesIO(json.dumps({'target_commitish':archives.STUDIES['exact']['commit'],
                        'prerelease':True,'draft':False,'html_url':'https://github.com/test'}).encode())
            with patch.object(archives,'github_token',return_value='inert-test-placeholder'),patch.object(archives.urllib.request,'build_opener',return_value=Opener()):
                archives.archive(raw,inputs,False,initialize_only=True)
            self.assertEqual(methods,['GET'])
            self.assertFalse((raw/'checkpoint-archive.json').exists())

    def test_ci_missing_release_cannot_create_source_tag(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            raw,inputs,_,_=self.prepared(folder);methods=[]
            class Opener:
                def open(self,request,timeout):
                    methods.append(request.get_method())
                    raise archives.urllib.error.HTTPError(request.full_url,404,'missing',None,io.BytesIO(b'{}'))
            with patch.object(archives,'github_token',return_value='inert-test-placeholder'),patch.object(archives.urllib.request,'build_opener',return_value=Opener()):
                with self.assertRaises(ValueError):archives.archive(raw,inputs,False,existing_only=True)
            self.assertEqual(methods,['GET'])


if __name__=='__main__':unittest.main()
