"""Resume pinned downloads with a shared rate limiter and HTTP backoff."""
import threading,time,urllib.error,urllib.request
from scripts.real_transfer_v19.acquire_deform360 import acquire

original=urllib.request.urlopen
lock=threading.Lock();next_request=0.

def limited(*args,**kwargs):
    global next_request
    for attempt in range(8):
        with lock:
            wait=max(0.,next_request-time.monotonic())
            next_request=max(next_request,time.monotonic())+.7
        if wait:time.sleep(wait)
        try:return original(*args,**kwargs)
        except urllib.error.HTTPError as error:
            if error.code not in (429,502,503,504):raise
            delay=max(float(error.headers.get('Retry-After','60')),min(300.,30.*2**attempt))
            with lock:next_request=max(next_request,time.monotonic()+delay)
            print('HTTP backoff',error.code,delay,flush=True)
    raise RuntimeError('Remote host still throttling after bounded retries')

if __name__=='__main__':
    urllib.request.urlopen=limited
    raise SystemExit(0 if acquire() else 1)
