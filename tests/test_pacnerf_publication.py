"""Reporting credentials and stationarity must not create false validation."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import download_study_artifact as artifact
    from pacnerf_comparison import stationarity_record
finally:sys.path.pop(0)


class PublicationTests(unittest.TestCase):
    def test_actions_credential_does_not_call_interactive_git(self):
        with patch.dict(os.environ,{'GITHUB_TOKEN':'inert-test-placeholder'}),patch.object(artifact.subprocess,'run') as run:
            self.assertEqual(artifact.github_token(),'inert-test-placeholder')
            run.assert_not_called()

    def test_local_credential_is_captured_without_printing(self):
        with patch.dict(os.environ,{'GITHUB_TOKEN':''}),patch.object(artifact.subprocess,'run',
            return_value=SimpleNamespace(returncode=0,stdout='username=test\npassword=inert-test-placeholder\n')) as run:
            self.assertEqual(artifact.github_token(),'inert-test-placeholder')
            self.assertTrue(run.call_args.kwargs['capture_output'])

    def test_stationary_flag_cannot_override_gradient_failure(self):
        record={'stationary':True,'gauss_newton_decrement':.01,
            'history':[{'objective':100.,'gradient_max_abs':.03}]}
        with self.assertRaises(ValueError):stationarity_record(record)


if __name__=='__main__':unittest.main()
