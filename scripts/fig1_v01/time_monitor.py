"""Non-invasive wall-clock logger for Figure-1 jobs: polls processes every 30 s and appends one JSON line per
finished job to runs/fig1-v01/timings.jsonl (pid, command, GPU, start/end, peak number of jobs sharing that GPU).
Resolution ~30 s; process start time is taken from /proc (exact)."""
import json, os, re, time
from pathlib import Path
OUT = Path('/home/bayespc2/vscode/guassianSDE/runs/fig1-v01/timings.jsonl')
PAT = re.compile(r'(pacnerf_stem\.py|gic_stem_entry\.py|gic_stem_predict\.py|mphygs/fit\.py|mphygs/forecast\.py|fem_control\.py|'
                 r'ours_\w+\.py|ours\.py|striker_truth\.py|block\.py|truth\.py|rgb_track\.py|block_rgb_track\.py)')
HZ = os.sysconf('SC_CLK_TCK'); BOOT = time.time()-float(open('/proc/uptime').read().split()[0])
live = {}


def scan():
    cur = {}
    for p in os.listdir('/proc'):
        if not p.isdigit():
            continue
        try:
            cmd = open(f'/proc/{p}/cmdline', 'rb').read().split(b'\0')
            args = [c.decode(errors='replace') for c in cmd if c]
            if not args or 'python' not in os.path.basename(args[0]) or not any(PAT.search(a) for a in args[1:3]):
                continue
            st = open(f'/proc/{p}/stat').read().rsplit(')', 1)[1].split()
            if int(st[1]) != 1 and os.path.exists(f'/proc/{st[1]}/cmdline') and open(f'/proc/{st[1]}/cmdline', 'rb').read() == open(f'/proc/{p}/cmdline', 'rb').read():
                continue  # multiprocessing worker of an already-tracked parent
            env = dict(e.split('=', 1) for e in open(f'/proc/{p}/environ', 'rb').read().decode(errors='replace').split('\0') if '=' in e)
            cur[int(p)] = dict(pid=int(p), cmd=' '.join(args[1:])[:600], gpu=env.get('CUDA_VISIBLE_DEVICES'),
                               start=BOOT+int(st[19])/HZ, unit=None)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return cur


while True:
    cur = scan(); now = time.time()
    for pid, j in cur.items():
        if pid not in live:
            live[pid] = dict(j, peak_jobs_on_gpu=1)
    for pid in list(live):
        j = live[pid]
        share = sum(1 for k in cur.values() if k['gpu'] == j['gpu']) if j['gpu'] not in (None, '') else 0
        j['peak_jobs_on_gpu'] = max(j['peak_jobs_on_gpu'], share)
        if pid not in cur:
            j['end'] = now; j['wall_s'] = j['end']-j['start']
            with OUT.open('a') as f:
                f.write(json.dumps(j)+'\n')
            del live[pid]
    time.sleep(30)
