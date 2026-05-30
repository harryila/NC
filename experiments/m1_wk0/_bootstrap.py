"""Make the pinned AKOrN checkout importable without manual PYTHONPATH.
Import this FIRST in any module that does `from source... import ...`.

Resolves AKOrN in this order: $AKORN_HOME, then <repo_root>/external/akorn.
Run setup.sh to populate external/akorn.
"""
import os
import sys


def _add_akorn():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    for cand in (os.environ.get("AKORN_HOME", ""),
                 os.path.join(repo_root, "external", "akorn")):
        if cand and os.path.isdir(os.path.join(cand, "source")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return cand
    raise ImportError(
        "AKOrN not found. Run ./setup.sh (clones it to external/akorn) or set AKORN_HOME=/path/to/akorn."
    )


AKORN_HOME = _add_akorn()
