"""Run frozen fitting/native code with the separately labelled local-grip adapter."""
import sys,runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.external_benchmark import data
from scripts.external_benchmark.contact_adapter import softvt
mode=sys.argv.pop(1)
if mode not in ['fit','native']:raise ValueError('Expected fit or native')
data.softvt=softvt
runpy.run_module('scripts.external_benchmark.'+('retry_fit' if mode=='fit' else 'native_softvt'),run_name='__main__')
