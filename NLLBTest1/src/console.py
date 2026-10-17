"""Console helpers shared by the scripts, so accents survive a cp1252 Windows terminal."""
import sys


def force_utf8(stream=None):
    for target in (sys.stdin, sys.stdout, sys.stderr) if stream is None else (stream,):
        try:
            target.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass