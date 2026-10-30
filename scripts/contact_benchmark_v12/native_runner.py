"""Run the unchanged native baseline adapter in v12 output/deadline scope."""
import sys,runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
# Bind original raw input paths BEFORE changing output globals in this process.
import scripts.external_benchmark.data
from scripts.contact_benchmark_v12 import common
import scripts.external_benchmark.common as original
original.BASE=common.BASE;original.check_deadline=common.check_deadline
original.save_provenance=common.provenance
runpy.run_module('scripts.external_benchmark.native_softvt',run_name='__main__')
