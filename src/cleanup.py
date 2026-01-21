import gc
import inspect
from typing import Iterable, Optional, Dict, Any

try:
    import psutil
except Exception:
    psutil = None


def force_cleanup(names: Optional[Iterable[str]] = None, global_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Delete named variables from `global_vars` (or caller globals) and run garbage collector.

    Returns a dict with list of deleted names and optional memory snapshot when `psutil` is available.
    """
    deleted = []
    if global_vars is None:
        frm = inspect.stack()[1].frame
        global_vars = frm.f_globals

    if names:
        for n in list(names):
            if n in global_vars:
                try:
                    del global_vars[n]
                    deleted.append(n)
                except Exception:
                    pass

    gc.collect()

    mem = {}
    if psutil is not None:
        proc = psutil.Process()
        mi = proc.memory_info()
        mem = {"rss": mi.rss, "vms": mi.vms}

    return {"deleted": deleted, "memory": mem}


def clear_common(global_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Delete a small set of commonly large variable names and trigger GC."""
    common = ["df", "data", "X", "y", "model", "pipeline", "pipe"]
    return force_cleanup(common, global_vars)


__all__ = ["force_cleanup", "clear_common"]
