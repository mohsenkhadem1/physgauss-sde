import os
import sys
from pathlib import Path
import tempfile
import unittest
import io
import hashlib
import json
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from archive_research_artifacts import archive
from restore_research_artifacts import restore,download_parts


class ArchiveTests(unittest.TestCase):
    def test_download_verifies_transport_and_never_keeps_corrupt_part(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=root/'manifest.json';parts=root/'parts'
            content=b'expected content'
            manifest.write_text(json.dumps({'parts':[{'name':'part-0000','bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}]}))
            with patch('urllib.request.urlopen',return_value=io.BytesIO(b'corrupt')):
                with self.assertRaisesRegex(ValueError,'checksum'):download_parts(manifest,'https://example.test/commit',parts)
            self.assertEqual(list(parts.iterdir()),[])
            with patch('urllib.request.urlopen',return_value=io.BytesIO(content)):
                self.assertEqual(download_parts(manifest,'https://example.test/commit',parts),1)
            self.assertEqual((parts/'part-0000').read_bytes(),content)

    def test_multipart_roundtrip_and_refuse_different_existing_file(self):
        previous=Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder);Path('evidence').mkdir()
                content=np.random.default_rng(7).integers(0,256,size=1_200_000,dtype=np.uint8).tobytes()
                Path('evidence/raw.bin').write_bytes(content);Path('evidence/log.txt').write_text('failedtrial preserved\n')
                result=archive(['evidence'],'parts','test',part_bytes=1_000_000)
                self.assertEqual(len(result['parts']),2)
                # Exercise assembly rather than reusing the archive just written.
                Path('parts/test.tar.gz').unlink()
                restored=restore('parts/manifest.json','parts','restored')
                self.assertEqual(restored['files_verified'],2)
                self.assertEqual(Path('restored/evidence/raw.bin').read_bytes(),content)
                Path('restored/evidence/log.txt').write_text('different')
                with self.assertRaises(FileExistsError):restore('parts/manifest.json','parts','restored')
            finally:os.chdir(previous)

    def test_symlink_input_is_rejected(self):
        previous=Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder);Path('source').write_text('data');Path('link').symlink_to('source')
                with self.assertRaises(ValueError):archive(['link'],'parts','test')
            finally:os.chdir(previous)


if __name__=='__main__':unittest.main()
