"""Tests of our metadata fingerprint; no upstream module is imported."""
import unittest
from inspect_bytecode import fingerprint

class FingerprintTests(unittest.TestCase):
    def test_debug_locations_ignored(self):
        a=compile("def f(x=3):\n return x+1\n", "a.py", "exec")
        b=compile("\n\ndef f(x=3):\n return x+1\n", "b.py", "exec")
        self.assertEqual(fingerprint(a),fingerprint(b))
    def test_default_change_detected(self):
        self.assertNotEqual(fingerprint(compile("def f(x=3): return x", "a", "exec")),fingerprint(compile("def f(x=4): return x", "a", "exec")))
    def test_nested_body_change_detected(self):
        self.assertNotEqual(fingerprint(compile("def f(x): return x+1", "a", "exec")),fingerprint(compile("def f(x): return x+2", "a", "exec")))

if __name__=="__main__":unittest.main()
