#!/usr/bin/env python3
"""Create or repair the local virtual environment.

This helper works on Windows, macOS, and Linux. It ensures that a `.venv`
virtual environment exists at the project root, upgrades pip, and installs
the dependencies listed in `requirements.txt`.

Usage:
    python scripts/bootstrap_env.py          # create or update .venv
    python scripts/bootstrap_env.py --force  # recreate the venv from scratch
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VENV = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
SUPPORTED_MIN = (3, 10)
SUPPORTED_MAX = (3, 12)


class BootstrapError(RuntimeError):
    """Raised when the environment bootstrap fails."""


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run a subprocess command and stream its output."""
    display = " ".join(command)
    print(f"→ {display}")
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise BootstrapError(f"Command failed (exit code {exc.returncode}): {display}") from exc


def resolve_python_executable(venv_path: Path) -> Path:
    scripts_dir = "Scripts" if platform.system().lower().startswith("win") else "bin"
    python_name = "python.exe" if platform.system().lower().startswith("win") else "python"
    return venv_path / scripts_dir / python_name


def ensure_requirements_file(path: Path) -> None:
    if not path.exists():
        raise BootstrapError(f"Missing requirements file: {path}")


def create_virtualenv(venv_path: Path, *, python_exe: str) -> None:
    python_cmd = python_exe
    print(f"Creating virtual environment at {venv_path} using {python_cmd}")
    run_command([python_cmd, "-m", "venv", str(venv_path)])


def upgrade_pip(venv_python: Path) -> None:
    run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])


def install_requirements(venv_python: Path, requirements_path: Path) -> None:
    run_command([str(venv_python), "-m", "pip", "install", "-r", str(requirements_path)])


def remove_virtualenv(venv_path: Path) -> None:
    if venv_path.exists():
        print(f"Removing existing virtual environment at {venv_path}")
        shutil.rmtree(venv_path)


def is_supported_version(major: int, minor: int) -> bool:
    return SUPPORTED_MIN <= (major, minor) <= SUPPORTED_MAX


def query_python_version(python_executable: Path) -> tuple[int, int]:
    try:
        result = subprocess.run(
            [str(python_executable), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise BootstrapError(f"Failed to query Python version for {python_executable}") from exc
    version_str = result.stdout.strip()
    major_str, minor_str = version_str.split(".")
    return int(major_str), int(minor_str)


def _discover_python_from_launcher(major: int, minor: int) -> str | None:
    """On Windows, try to resolve `py -3.x` to a concrete interpreter path."""
    try:
        completed = subprocess.run(
            ["py", f"-{major}.{minor}", "-c", "import sys; print(sys.executable)"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    interpreter = completed.stdout.strip()
    return interpreter or None


def select_python_interpreter(requested: str | None) -> str:
    if requested:
        return requested

    current = sys.version_info
    if current.major == 3 and is_supported_version(current.major, current.minor):
        return sys.executable

    preferred_versions = [(3, 12), (3, 11), (3, 10)]

    for major, minor in preferred_versions:
        candidate = shutil.which(f"python{major}.{minor}")
        if candidate:
            print(f"Detected Python {major}.{minor} at {candidate}")
            return candidate

    if platform.system().lower().startswith("win"):
        for major, minor in preferred_versions:
            interpreter = _discover_python_from_launcher(major, minor)
            if interpreter:
                print(f"Detected Python {major}.{minor} via py launcher at {interpreter}")
                return interpreter

    raise BootstrapError(
        "Python 3.13+ detected, but the pinned dependencies currently expect Python 3.10–3.12.\n"
        "Install Python 3.11 (recommended) and rerun with `python3.11 scripts/bootstrap_env.py`\n"
        "on Linux/macOS or `py -3.11 scripts\\bootstrap_env.py` on Windows."
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the local virtual environment.")
    parser.add_argument(
        "--venv",
        type=Path,
        default=DEFAULT_VENV,
        help="Path to the virtual environment (default: %(default)s)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python interpreter to use when creating the environment (defaults to current interpreter).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate the virtual environment from scratch even if it already exists.",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=REQUIREMENTS_FILE,
        help="Requirements file to install (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    ensure_requirements_file(args.requirements)

    python_cmd = select_python_interpreter(args.python)

    if args.force:
        remove_virtualenv(args.venv)

    if not args.venv.exists():
        create_virtualenv(args.venv, python_exe=python_cmd)
    else:
        print(f"Virtual environment already exists at {args.venv}, will update dependencies.")

    venv_python = resolve_python_executable(args.venv)
    recreate_needed = False
    if not venv_python.exists():
        print("Virtual environment appears broken (missing python). Recreating it.")
        recreate_needed = True
    else:
        major, minor = query_python_version(venv_python)
        if not is_supported_version(major, minor):
            print(
                f"Existing virtualenv uses Python {major}.{minor}, "
                "which is outside the supported range. Recreating it."
            )
            recreate_needed = True

    if recreate_needed:
        remove_virtualenv(args.venv)
        create_virtualenv(args.venv, python_exe=python_cmd)
        venv_python = resolve_python_executable(args.venv)
        if not venv_python.exists():
            raise BootstrapError(f"Could not find Python executable inside the virtualenv: {venv_python}")

    upgrade_pip(venv_python)
    install_requirements(venv_python, args.requirements)

    if platform.system().lower().startswith("win"):
        activation = args.venv / "Scripts" / "Activate.ps1"
    else:
        activation = args.venv / "bin" / "activate"

    print("\nEnvironment ready! 👇")
    if platform.system().lower().startswith("win"):
        print(f"  PowerShell: {activation}")
        print(f"  CMD:        {args.venv / 'Scripts' / 'activate.bat'}")
    else:
        print(f"  Activate: source {activation}")
    print(f"\nLaunch app: {venv_python} pyqt_app.py")


if __name__ == "__main__":
    try:
        main()
    except BootstrapError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(130)
