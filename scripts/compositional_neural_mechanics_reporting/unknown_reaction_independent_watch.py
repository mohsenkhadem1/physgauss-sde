"""Read-only bounded waiter, no study restart/replay/scoring."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';B=ROOT/'runs/compositional-neural-mechanics-v16/unknown-action-reserved';deadline=json.loads((B/'stage-clock.json').read_text())['deadline_unix']+2400
while time.time()<deadline:
 if (R/'unknown-reaction-secondary-results.json').exists():
  r=subprocess.run([sys.executable,'-m','scripts.compositional_neural_mechanics_reporting.unknown_reaction_independent_audit'],cwd=ROOT,timeout=max(.1,min(180,deadline-time.time())));raise SystemExit(r.returncode)
 time.sleep(min(60,max(0,deadline-time.time())))
raise SystemExit('Reporting deadline reached without terminal secondary results; no restart')
