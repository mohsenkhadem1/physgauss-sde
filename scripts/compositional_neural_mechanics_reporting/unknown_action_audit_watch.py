"""Bounded read-only watcher; never starts/restarts the study or scorer."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'runs/compositional-neural-mechanics-v16/unknown-action-reserved';R=ROOT/'reports/compositional-neural-mechanics-v16'
deadline=json.loads((B/'stage-clock.json').read_text())['deadline_unix']+120
while time.time()<deadline:
 if (B/'fits-terminal.json').exists()and(R/'unknown-action-reserved-results.json').exists():
  result=subprocess.run([sys.executable,str(ROOT/'scripts/compositional_neural_mechanics_reporting/unknown_action_independent_audit.py')],cwd=ROOT,timeout=max(.1,min(120,deadline-time.time())))
  raise SystemExit(result.returncode)
 time.sleep(min(60,max(0,deadline-time.time())))
raise SystemExit('Audit deadline reached without complete terminal and primary score reports; no study restarted.')
