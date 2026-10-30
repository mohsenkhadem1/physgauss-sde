"""Stop only v12 registered process identities at the absolute campaign limit."""
import sys, time, os, signal, json
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.contact_benchmark_v12.common import BASE, CONFIG, write_json
deadline=datetime.fromisoformat(json.loads(CONFIG.read_text())['deadline_utc'].replace('Z','+00:00')).timestamp()
while time.time()<deadline:
    time.sleep(min(30,max(0,deadline-time.time())))
stopped=[]
for path in (BASE/'pids').glob('*.json'):
    record=json.loads(path.read_text());pid=record['pid']
    try:
        ticks=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
        if ticks==record['start_ticks']:
            os.kill(pid,signal.SIGTERM);stopped.append(pid)
    except (FileNotFoundError,ProcessLookupError):pass
write_json(BASE/'deadline-reached.json',dict(stopped=stopped,time=datetime.now(timezone.utc).isoformat()))
