#!/usr/bin/env python3
"""Docker management utility for the YOPO project."""

import os
import re
import json
import grp
import shutil
import socket
import getpass
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Union


def get_hostname() -> str:
    return socket.gethostname()


def is_user_in_docker_group() -> bool:
    try:
        docker_gid = grp.getgrnam("docker").gr_gid
        return docker_gid in os.getgroups()
    except KeyError:
        return False


class YOPOContainer:
    """Manage Docker containers for the YOPO project."""

    IMAGE_NAME = "yopo"
    IMAGE_TAG = "cu118"
    CONTAINER_WORKSPACE = "/root/workspace"

    def __init__(self, project_dir: Path, mount_dir: Path):
        self.project_dir = project_dir.resolve()
        self.mount_dir = mount_dir.resolve()
        self.docker_dir = self.project_dir / "docker"
        self.hostname = get_hostname()
        self.container_name = f"yopo-{self.project_dir.name}"

        assert shutil.which("docker"), (
            "Docker is not installed! "
            "Install via https://docs.docker.com/engine/install/ubuntu/"
        )
        assert is_user_in_docker_group(), (
            f"Current user is not in the 'docker' group. Run:\n"
            f"  sudo usermod -aG docker {getpass.getuser()}\n"
            f"then log out and back in."
        )

    @property
    def image_full(self) -> str:
        return f"{self.IMAGE_NAME}:{self.IMAGE_TAG}"

    # ------------------------------------------------------------------
    # Docker queries
    # ------------------------------------------------------------------

    def _docker_output(self, *args) -> str:
        return subprocess.run(
            ["docker", *args],
            capture_output=True, text=True, check=False,
        ).stdout.strip()

    def does_image_exist(self) -> bool:
        images = self._docker_output("images", "--format", "{{.Repository}}:{{.Tag}}")
        return self.image_full in images.splitlines()

    def get_running_container(self) -> Optional[str]:
        names = self._docker_output("ps", "--format", "{{.Names}}")
        for name in names.splitlines():
            if self.container_name in name:
                return name.strip()
        return None

    def has_nvidia_runtime(self) -> bool:
        runtimes_raw = self._docker_output("info", "--format", "{{json .Runtimes}}")
        if not runtimes_raw:
            return False
        try:
            runtimes = json.loads(runtimes_raw)
        except json.JSONDecodeError:
            return False
        return "nvidia" in runtimes

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def build(self):
        """Build the Docker image from the Dockerfile."""
        if self.does_image_exist():
            print(f"[INFO] Image '{self.image_full}' already exists. Rebuilding...")

        proxy_args = []
        for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            val = os.environ.get(var, "")
            if val:
                proxy_args += ["--build-arg", f"{var}={val}"]
                print(f"[INFO] Using proxy: {var}={val}")

        cmd = [
            "docker", "build",
            "--network=host",
            "-t", self.image_full,
            "-f", str(self.docker_dir / "Dockerfile"),
            *proxy_args,
            str(self.project_dir),
        ]
        print(f"[INFO] Building image '{self.image_full}' ...")
        subprocess.run(cmd, check=True)
        print(f"[INFO] Image '{self.image_full}' built successfully.")

    def start(self):
        """Start the container in detached mode."""
        running = self.get_running_container()
        if running:
            print(f"[INFO] Container '{running}' is already running.")
            return

        if not self.does_image_exist():
            raise RuntimeError(
                f"Image '{self.image_full}' not found. Run `python docker.py build` first."
            )

        display = os.environ.get("DISPLAY", ":0")
        xauth = Path("~/.Xauthority").expanduser()

        cmd = [
            "docker", "run",
            "--rm", "-dit",
            "--gpus", "all",
            "--name", self.container_name,
            "--hostname", self.hostname,
            # X11 forwarding
            "--mount", f"type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix",
            "--mount", f"type=bind,source={xauth},target=/root/.Xauthority",
            # mount the whole workspace (e.g. /home/wrq/workspace/fm-gcopter-dataset)
            "--mount", f"type=bind,source={self.mount_dir},target={self.CONTAINER_WORKSPACE}",
            # environment
            f"--env=DISPLAY={display}",
            f"--env=NVIDIA_VISIBLE_DEVICES=all",
            f"--env=NVIDIA_DRIVER_CAPABILITIES=all",
            f"--env=__NV_PRIME_RENDER_OFFLOAD=1",
            f"--env=__GLX_VENDOR_LIBRARY_NAME=nvidia",
            f"--env=__VK_LAYER_NV_optimus=NVIDIA_only",
            f"--env=TZ=Asia/Shanghai",
            f"--env=FLIGHTMARE_PATH={self.CONTAINER_WORKSPACE}/YOPO",
            f"--env=PYTHONPATH={self.CONTAINER_WORKSPACE}/YOPO:{self.CONTAINER_WORKSPACE}/YOPO/flightlib/build",
            "--network=host",
            self.image_full,
        ]

        # Newer Docker setups can use --gpus without a named nvidia runtime.
        # Keep runtime flag only when it is actually registered.
        if self.has_nvidia_runtime():
            cmd.insert(6, "--runtime=nvidia")
        else:
            print("[INFO] Docker runtime 'nvidia' not found; using '--gpus all' only.")

        print(f"[INFO] Starting container '{self.container_name}' (DISPLAY={display}) ...")
        subprocess.run(cmd, check=True)
        print(f"[INFO] Container '{self.container_name}' started.")

    def enter(self):
        """Exec into the running container."""
        running = self.get_running_container()
        if not running:
            raise RuntimeError(
                f"No running container found for '{self.container_name}'. "
                f"Run `python docker.py start` first."
            )

        display = os.environ.get("DISPLAY", ":0")
        print(f"[INFO] Entering container '{running}' ...\n")
        subprocess.run([
            "docker", "exec", "-it",
            "--workdir", f"{self.CONTAINER_WORKSPACE}/YOPO/run",
            f"--env=DISPLAY={display}",
            running,
            "bash",
        ])

    def stop(self):
        """Stop the running container."""
        running = self.get_running_container()
        if not running:
            raise RuntimeError(f"No running container found for '{self.container_name}'.")

        print(f"[INFO] Stopping container '{running}' ...")
        subprocess.run(["docker", "stop", running], check=False)
        print(f"[INFO] Container '{running}' stopped.")

    def logs(self):
        """Show container logs."""
        running = self.get_running_container()
        if not running:
            raise RuntimeError(f"No running container found for '{self.container_name}'.")
        subprocess.run(["docker", "logs", "--tail", "100", "-f", running])

    def status(self):
        """Print the status of image and container."""
        img = "EXISTS" if self.does_image_exist() else "NOT FOUND"
        running = self.get_running_container()
        ctr = running if running else "NOT RUNNING"
        print(f"  Image:     {self.image_full}  [{img}]")
        print(f"  Container: {self.container_name}  [{ctr}]")


def main():
    project_dir = Path(__file__).resolve().parent.parent
    default_mount = project_dir.parent  # /home/wrq/workspace/fm-gcopter-dataset

    parser = argparse.ArgumentParser(
        description="YOPO Docker management utility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python docker.py build          # Build the image\n"
            "  python docker.py start          # Start container (detached)\n"
            "  python docker.py enter          # Exec bash into container\n"
            "  python docker.py stop           # Stop the container\n"
            "  python docker.py status         # Show image/container status\n"
            "  python docker.py logs           # Tail container logs\n"
        ),
    )
    parser.add_argument(
        "-d", "--mount-dir",
        default=str(default_mount),
        help=f"Host directory to mount into the container (default: {default_mount})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build",  help="Build the Docker image from Dockerfile.")
    subparsers.add_parser("start",  help="Start the container in detached mode.")
    subparsers.add_parser("enter",  help="Exec into the running container.")
    subparsers.add_parser("stop",   help="Stop the running container.")
    subparsers.add_parser("status", help="Show image and container status.")
    subparsers.add_parser("logs",   help="Tail the container logs.")

    args = parser.parse_args()
    ci = YOPOContainer(project_dir=project_dir, mount_dir=Path(args.mount_dir))

    dispatch = {
        "build":  ci.build,
        "start":  ci.start,
        "enter":  ci.enter,
        "stop":   ci.stop,
        "status": ci.status,
        "logs":   ci.logs,
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
