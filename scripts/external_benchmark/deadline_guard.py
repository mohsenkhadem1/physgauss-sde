"""At the absolute deadline, stop only registered campaign PIDs with matching identity."""
from pathlib import Path
import sys,time,os,signal,json
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.external_benchmark.common import CONFIG,BASE,write_json
spec=json.loads(CONFIG.read_text());deadline=datetime.fromisoformat(spec['deadline_utc'].replace('Z','+00:00')).timestamp()
while time.time()<deadline:time.sleep(min(30,max(0,deadline-time.time())))
stopped=[]
for path in (BASE/'pids').glob('*.json'):
    record=json.loads(path.read_text());pid=record['pid']
    try:
        start=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if start==record['start_ticks'] and 'scripts/external_benchmark/' in cmd:
            os.kill(pid,signal.SIGTERM);stopped.append(pid)
    except (FileNotFoundError,ProcessLookupError):pass
write_json(BASE/'deadline-reached.json',dict(stopped=stopped,time=datetime.now(timezone.utc).isoformat()))
