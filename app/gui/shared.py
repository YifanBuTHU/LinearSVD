import ctypes
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_conda_executable():
    candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda.exe"),
        shutil.which("conda"),
        shutil.which("conda.bat"),
    ]
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if path.suffix.lower() in {".bat", ".cmd"}:
                exe_candidate = path.parents[1] / "Scripts" / "conda.exe"
                if exe_candidate.exists():
                    return str(exe_candidate)
            return str(path)
    return "conda"


def enable_high_dpi_awareness():
    if os.name != "nt":
        return

    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass

    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def configure_tk_scaling(root):
    try:
        dpi = float(root.winfo_fpixels("1i"))
    except Exception:
        return 1.0

    scaling = max(dpi / 72.0, 1.0)
    try:
        root.tk.call("tk", "scaling", scaling)
    except Exception:
        pass
    return scaling
