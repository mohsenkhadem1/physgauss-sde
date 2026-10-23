"""Owned process status reads: tolerate normal exit races only."""
import errno
from pathlib import Path


def proc_text(pid,name):
 try:return Path('/proc',str(pid),name).read_text()
 except OSError as error:
  if error.errno in (errno.ENOENT,errno.ESRCH):return None
  raise


def proc_rss(pid):
 text=proc_text(pid,'status')
 if text is None:return 0
 return next((int(x.split()[1])*1024 for x in text.splitlines()if x.startswith('VmRSS:')),0)


def proc_identity(pid):
 text=proc_text(pid,'stat')
 if text is None:return None
 fields=text[text.rfind(')')+2:].split();return int(fields[1]),fields[19]
