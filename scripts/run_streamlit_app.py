from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "app" / "main.py"


def main() -> int:
    command = [sys.executable, "-m", "streamlit", "run", str(APP_PATH), *sys.argv[1:]]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        creationflags=creationflags,
    )

    def stop_process(_signum=None, _frame=None) -> None:
        if process.poll() is not None:
            raise SystemExit(process.returncode or 0)
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        raise SystemExit(0)

    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is None:
            continue
        try:
            signal.signal(signal_value, stop_process)
        except (ValueError, OSError):
            continue

    try:
        return process.wait()
    except KeyboardInterrupt:
        stop_process()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
