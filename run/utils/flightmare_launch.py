import atexit
import os
import signal
import subprocess
import time


def build_flightmare_env():
    """Prefer NVIDIA PRIME offload when an X display is available."""
    env = os.environ.copy()
    if env.get("DISPLAY"):
        env.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
        env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
        env.setdefault("__VK_LAYER_NV_optimus", "NVIDIA_only")
    return env


def cleanup_stale_flightmare(binary_name: str = "flightmare.x86_64"):
    """Kill stale Flightmare processes to avoid connection conflicts across reruns."""
    result = subprocess.run(
        ["pgrep", "-f", binary_name],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)

    if not pids:
        return

    print(f"[INFO] terminating stale Flightmare processes: {pids}", flush=True)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.time() + 5.0
    while time.time() < deadline:
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                pass
        if not alive:
            return
        time.sleep(0.2)

    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def terminate_process(proc):
    if proc is None:
        return
    try:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def launch_flightmare(binary_path: str):
    """Launch Flightmare in the background with NVIDIA offload hints when available."""
    cleanup_stale_flightmare(os.path.basename(binary_path))
    env = build_flightmare_env()
    launch_env = {
        "DISPLAY": env.get("DISPLAY", ""),
        "__NV_PRIME_RENDER_OFFLOAD": env.get("__NV_PRIME_RENDER_OFFLOAD", ""),
        "__GLX_VENDOR_LIBRARY_NAME": env.get("__GLX_VENDOR_LIBRARY_NAME", ""),
        "__VK_LAYER_NV_optimus": env.get("__VK_LAYER_NV_optimus", ""),
    }
    print(f"[INFO] launching Flightmare with env={launch_env}", flush=True)
    proc = subprocess.Popen(
        [binary_path],
        cwd=os.path.dirname(binary_path),
        env=env,
        start_new_session=True,
    )
    atexit.register(terminate_process, proc)
    return proc
