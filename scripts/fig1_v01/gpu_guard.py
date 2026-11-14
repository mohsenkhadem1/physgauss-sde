"""GPU health guard (user-level, no root needed). Added after the 2026-11-10 00:16 fault: GPU 0000:67:00 "fell off the
bus" (Xid 79) under load and the driver demanded a node reboot; the queues then burned through their steps failing.

  * every POLL_S seconds: per-GPU telemetry -> runs/gpu-health/telemetry-YYYYMMDD.csv
    (temp, power draw/limit, SM clock, throttle reasons, PCIe gen/width, memory, fan)
  * follows the kernel log; on a fatal NVRM Xid (79 fallen off bus, 154 reboot required, 119/120 GSP timeout,
    48 DBE, 62/45-storms are not fatal) it stops every job unit (fig1-*, mphygs-*, ptw-*, blk-*, stem-*, tcnn-*),
    writes runs/gpu-health/FAULT-<ts>.json with the Xid lines and the last 10 min of telemetry, and sends a desktop
    notification.  Steps are idempotent, so the queues resume cleanly with resume_after_reboot.sh.
  * warnings (temp >= 84 C, PCIe link width < 16, power/thermal slowdown) -> runs/gpu-health/warnings.log
usage: python3 gpu_guard.py            (run as a capped user unit; see resume_after_reboot.sh)
"""
import collections, csv, json, re, subprocess, threading, time
from datetime import datetime
from pathlib import Path

R = Path(__file__).resolve().parents[2]
OUT = R/'runs/gpu-health'; OUT.mkdir(parents=True, exist_ok=True)
POLL_S = 15
FATAL = {'79', '154', '119', '120', '48', '74', '94', '95'}
JOB_PATTERNS = ['fig1-q*', 'mphygs-*', 'ptw-*', 'blk-*', 'stem-*', 'tcnn-*']
Q = ('index,pci.bus_id,temperature.gpu,power.draw,power.limit,clocks.sm,clocks_event_reasons.active,'
     'pcie.link.gen.current,pcie.link.width.current,memory.used,fan.speed,utilization.gpu')
FIELDS = ['time'] + Q.split(',')
recent = collections.deque(maxlen=10*60//POLL_S*4)
lock = threading.Lock()
tripped = threading.Event()


def warn(msg):
    with open(OUT/'warnings.log', 'a') as f:
        f.write(f'{datetime.now().isoformat(timespec="seconds")} {msg}\n')


def notify(msg):
    subprocess.run(['notify-send', '-u', 'critical', 'GPU guard', msg], capture_output=True)


def poll():
    while True:
        ts = datetime.now().isoformat(timespec='seconds')
        try:
            out = subprocess.run(['nvidia-smi', f'--query-gpu={Q}', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=30).stdout
        except Exception as e:  # nvidia-smi hanging is itself a symptom
            out = ''; warn(f'nvidia-smi failed: {e!r}')
        rows = [[ts] + [c.strip() for c in ln.split(',')] for ln in out.strip().splitlines() if ln.strip()]
        if len(rows) != 4:
            warn(f'nvidia-smi returned {len(rows)} GPUs: {out.strip()[:300]!r}')
        p = OUT/f'telemetry-{datetime.now():%Y%m%d}.csv'; new = not p.exists()
        with open(p, 'a', newline='') as f:
            w = csv.writer(f)
            if new: w.writerow(FIELDS)
            w.writerows(rows)
        with lock:
            recent.extend(rows)
        for r in rows:
            d = dict(zip(FIELDS, r))
            try:
                if float(d['temperature.gpu']) >= 84: warn(f"GPU{d['index']} temp {d['temperature.gpu']} C")
                if d['pcie.link.width.current'] not in ('16', '[N/A]'): warn(f"GPU{d['index']} PCIe width x{d['pcie.link.width.current']}")
                reasons = int(d['clocks_event_reasons.active'], 16)
                if reasons & 0x4 or reasons & 0x20 or reasons & 0x40 or reasons & 0x80:  # sw power cap / sw-hw thermal / hw slowdown
                    if reasons & 0x88: warn(f"GPU{d['index']} HW slowdown/brake reasons=0x{reasons:x} power={d['power.draw']}W")
            except ValueError:
                pass
        time.sleep(POLL_S)


def trip(lines):
    if tripped.is_set():
        return
    tripped.set()
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    units = subprocess.run(['systemctl', '--user', 'list-units', '--plain', '--no-legend', '--state=active', *JOB_PATTERNS],
                           capture_output=True, text=True).stdout.split()
    units = [u for u in units if u.endswith('.service')]
    if units:
        subprocess.run(['systemctl', '--user', 'stop', *units], capture_output=True)
    with lock:
        tel = [dict(zip(FIELDS, r)) for r in recent]
    (OUT/f'FAULT-{ts}.json').write_text(json.dumps(dict(xid_lines=lines, stopped_units=units, telemetry_last_10min=tel), indent=1))
    warn(f'FATAL Xid -> stopped {units}; see FAULT-{ts}.json')
    notify(f'GPU fault (Xid). Stopped {len(units)} jobs. Reboot, then run scripts/fig1_v01/resume_after_reboot.sh')


def follow_kernel():
    p = subprocess.Popen(['journalctl', '-k', '-f', '-n', '0', '-o', 'short-iso'], stdout=subprocess.PIPE, text=True)
    buf = []
    for ln in p.stdout:
        m = re.search(r'NVRM: Xid \((PCI:[^)]*)\): (\d+)', ln)
        if m:
            buf.append(ln.strip()); warn(ln.strip())
            if m.group(2) in FATAL:
                time.sleep(2)  # let the burst of related lines land
                trip(buf[-20:])
        elif 'fallen off the bus' in ln:
            buf.append(ln.strip()); trip(buf[-20:])


if __name__ == '__main__':
    threading.Thread(target=poll, daemon=True).start()
    follow_kernel()
