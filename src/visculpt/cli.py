"""Command-line interface for installing and running ViSculpt."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .sam3_torch_setup import (
    CPU_BACKEND,
    TORCH_VERSION,
    TORCHVISION_VERSION,
    TorchBackend,
    WindowsGpu,
    detect_windows_gpu,
    format_probe,
    probe_torch_installation,
    select_windows_backend,
    torch_install_command,
)
from .workflow import SculptAgentWorkflow, SculptWorkflowError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web"
SAM3_ROOT = PROJECT_ROOT / "dependencies" / "sam3-inference-service"
ADDON_ROOT = (
    PROJECT_ROOT / "dependencies" / "Geometry-Editing-Blender-Add-on"
)
SAM3_LOCAL_CHECKPOINT = SAM3_ROOT / "checkpoint" / "sam3.pt"
_TERMINAL_EXIT_REQUESTED = False


class CliError(RuntimeError):
    """Report an actionable command-line failure."""


def _required_command(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise CliError(f"Required command '{name}' was not found on PATH.")
    return executable


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    print(f"Running: {' '.join(command)}", flush=True)
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    except subprocess.CalledProcessError as error:
        raise CliError(
            f"Command failed with exit code {error.returncode}: "
            f"{' '.join(command)}"
        ) from error


def _setup_main_environment() -> None:
    uv = _required_command("uv")
    npm = _required_command("npm")
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        shutil.copyfile(PROJECT_ROOT / ".env.example", env_path)
        print("Created .env from .env.example.")
    else:
        print("Kept the existing .env file.")
    _run([uv, "sync", "--locked"], cwd=PROJECT_ROOT)
    _run(
        [
            npm,
            "ci",
            "--cache",
            str(PROJECT_ROOT / ".cache" / "npm"),
        ],
        cwd=WEB_ROOT,
    )
    _run([npm, "run", "build"], cwd=WEB_ROOT)
    print("ViSculpt is configured. Add the required LLM API key to .env.")


def _setup_sam3_environment() -> None:
    uv = _required_command("uv")
    child_env = os.environ.copy()
    child_env.pop("VIRTUAL_ENV", None)
    child_env["PYTHONUTF8"] = "1"
    selection = None
    if os.name == "nt":
        selection = select_windows_backend(detect_windows_gpu())
        print(f"PyTorch backend selection: {selection.reason}")
    _run(
        [
            uv,
            "sync",
            "--project",
            str(SAM3_ROOT),
            "--locked",
            "--no-install-package",
            "torch",
            "--no-install-package",
            "torchvision",
        ],
        cwd=PROJECT_ROOT,
        env=child_env,
    )
    if selection is not None:
        _configure_windows_sam3_torch(
            uv,
            selection.backend,
            selection.gpu,
            child_env,
        )
    else:
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(_sam3_environment_python()),
                "--reinstall",
                "--no-deps",
                f"torch=={TORCH_VERSION}",
                f"torchvision=={TORCHVISION_VERSION}",
            ],
            cwd=SAM3_ROOT,
            env=child_env,
        )
    downloader = _sam3_environment_executable("sam3-download-checkpoint")
    if not downloader.is_file():
        raise CliError(
            "The SAM 3 checkpoint installer is missing after environment setup."
        )
    _run(
        [str(downloader), "--output", str(SAM3_LOCAL_CHECKPOINT)],
        cwd=SAM3_ROOT,
        env=child_env,
    )
    print(
        "SAM 3 is configured with a local checkpoint. Runtime downloads are "
        "disabled."
    )


def _configure_windows_sam3_torch(
    uv: str,
    backend: TorchBackend,
    gpu: WindowsGpu | None,
    child_env: dict[str, str],
) -> None:
    """Install, verify, and if necessary replace a Windows CUDA build."""

    selected_gpu = gpu if backend.cuda_version is not None else None
    try:
        _install_and_verify_sam3_torch(
            uv,
            backend,
            selected_gpu,
            child_env,
        )
        return
    except (CliError, RuntimeError) as error:
        if backend.cuda_version is None:
            raise CliError(str(error)) from error
        print(
            f"CUDA backend {backend.name} failed validation: {error}\n"
            "Falling back to the CPU backend.",
            file=sys.stderr,
        )
    try:
        _install_and_verify_sam3_torch(
            uv,
            CPU_BACKEND,
            None,
            child_env,
        )
    except (CliError, RuntimeError) as error:
        raise CliError(f"CPU backend fallback failed: {error}") from error


def _install_and_verify_sam3_torch(
    uv: str,
    backend: TorchBackend,
    gpu: WindowsGpu | None,
    child_env: dict[str, str],
) -> None:
    python = _sam3_environment_python()
    _run(
        torch_install_command(uv, python, backend),
        cwd=SAM3_ROOT,
        env=child_env,
    )
    probe = probe_torch_installation(python, gpu, env=child_env)
    print(format_probe(probe))
    failures = probe.failures_for(backend)
    if failures:
        raise CliError("; ".join(failures))
    print(f"Selected SAM 3 PyTorch backend: {backend.name}")


def _find_blender(explicit_path: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path.expanduser())
    configured = os.getenv("BLENDER_EXECUTABLE", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    on_path = shutil.which("blender")
    if on_path:
        candidates.append(Path(on_path))
    if sys.platform == "darwin":
        candidates.append(
            Path("/Applications/Blender.app/Contents/MacOS/Blender")
        )
    elif os.name == "nt":
        program_files = os.getenv("PROGRAMFILES")
        if program_files:
            root = Path(program_files) / "Blender Foundation"
            candidates.extend(
                sorted(root.glob("Blender */blender.exe"), reverse=True)
            )
            candidates.append(root / "Blender" / "blender.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise CliError(
        "Blender was not found. Install Blender 4.2 or newer, set "
        "BLENDER_EXECUTABLE, or pass --blender PATH."
    )


def _install_addon(blender_path: Path | None) -> None:
    blender = _find_blender(blender_path)
    build_script = ADDON_ROOT / "scripts" / "build_extension.py"
    _run([sys.executable, str(build_script)], cwd=PROJECT_ROOT)
    archive = ADDON_ROOT / "dist" / "geometry_editing_rpc-1.0.0.zip"
    if not archive.is_file():
        raise CliError(f"The Add-on archive was not created: {archive}")
    _run(
        [
            str(blender),
            "--background",
            "--command",
            "extension",
            "install-file",
            "-r",
            "user_default",
            str(archive),
        ],
        cwd=PROJECT_ROOT,
    )
    print(
        "The Geometry Editing RPC Add-on is installed. Enable it manually "
        "in Blender Preferences > Extensions."
    )


def _sam3_environment_executable(name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return (
        SAM3_ROOT
        / ".venv"
        / ("Scripts" if os.name == "nt" else "bin")
        / f"{name}{suffix}"
    )


def _sam3_environment_python() -> Path:
    executable = _sam3_environment_executable("python")
    if not executable.is_file():
        raise CliError(
            "The SAM 3 Python interpreter is missing after environment setup."
        )
    return executable


def _sam3_executable() -> Path:
    executable = _sam3_environment_executable("sam3-gradio")
    if not executable.is_file():
        raise CliError(
            "The SAM 3 environment is missing. Run "
            "'uv run visculpt setup-sam3' first."
        )
    return executable


def _sam3_checkpoint() -> Path:
    configured = os.getenv("SAM3_CHECKPOINT", "").strip()
    checkpoint = (
        Path(configured).expanduser() if configured else SAM3_LOCAL_CHECKPOINT
    )
    if not checkpoint.is_file():
        raise CliError(
            "The local SAM 3 checkpoint is missing. Run "
            "'uv run visculpt setup-sam3' first."
        )
    return checkpoint.resolve()


def _start_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    print(f"Starting: {' '.join(command)}", flush=True)
    # Inherit the launcher's terminal session and foreground process group.
    return subprocess.Popen(command, cwd=cwd, env=env)


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            process.terminate()
        process.wait(timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass


def _handle_terminal_exit(_signum: int, _frame: object) -> None:
    """Start cleanup once and ignore repeated terminal exit signals."""
    global _TERMINAL_EXIT_REQUESTED
    if _TERMINAL_EXIT_REQUESTED:
        return
    _TERMINAL_EXIT_REQUESTED = True
    raise KeyboardInterrupt


def _install_terminal_exit_handlers() -> dict[int, object]:
    """Install temporary handlers for terminal and host-process shutdown."""
    global _TERMINAL_EXIT_REQUESTED
    _TERMINAL_EXIT_REQUESTED = False
    previous_handlers: dict[int, object] = {}
    for signal_name in ("SIGHUP", "SIGTERM", "SIGBREAK", "SIGINT"):
        exit_signal = getattr(signal, signal_name, None)
        if exit_signal is None:
            continue
        previous_handlers[int(exit_signal)] = signal.signal(
            exit_signal,
            _handle_terminal_exit,
        )
    return previous_handlers


def _restore_signal_handlers(previous_handlers: dict[int, object]) -> None:
    """Restore signal handlers after the service supervisor exits."""
    for signal_number, handler in previous_handlers.items():
        signal.signal(signal_number, handler)


def _wait_until_ready(
    checks: dict[str, tuple[str, bytes]],
    processes: list[subprocess.Popen[bytes]],
    *,
    timeout_seconds: float = 180.0,
) -> None:
    pending = dict(checks)
    deadline = time.monotonic() + timeout_seconds
    while pending and time.monotonic() < deadline:
        for process in processes:
            return_code = process.poll()
            if return_code is not None:
                raise CliError(
                    f"A ViSculpt service exited early with code {return_code}."
                )
        for name, (url, marker) in list(pending.items()):
            if _url_is_ready(url, marker):
                print(f"Ready: {name} ({url})")
                pending.pop(name)
        if pending:
            time.sleep(0.25)
    if pending:
        missing = ", ".join(pending)
        raise CliError(f"Timed out while waiting for: {missing}.")


def _url_is_ready(url: str, marker: bytes) -> bool:
    try:
        with urlopen(url, timeout=1.0) as response:
            body = response.read(65_536)
            return response.status < 500 and marker in body
    except (OSError, URLError):
        return False


def _start_services(*, open_browser: bool) -> None:
    if not (WEB_ROOT / ".next" / "BUILD_ID").is_file():
        raise CliError(
            "The Web app has not been built. Run 'uv run visculpt setup' first."
        )
    langgraph = _required_command("langgraph")
    npm = _required_command("npm")
    sam3 = _sam3_executable()
    sam3_checkpoint = _sam3_checkpoint()
    python_service_env = os.environ.copy()
    python_service_env["PYTHONUTF8"] = "1"
    agent_server_env = python_service_env.copy()
    agent_server_env["LOG_LEVEL"] = "ERROR"
    services = [
        (
            "SAM 3",
            ("http://127.0.0.1:7860/gradio_api/info", b'"/segment"'),
            [
                str(sam3),
                "--device",
                "auto",
                "--checkpoint",
                str(sam3_checkpoint),
                "--no-download",
            ],
            SAM3_ROOT,
            python_service_env,
        ),
        (
            "Agent Server",
            ("http://127.0.0.1:2024/healthz", b"workflow-http-app"),
            [
                langgraph,
                "dev",
                "--no-browser",
                "--no-reload",
                "--n-jobs-per-worker",
                "1",
                "--allow-blocking",
                "--server-log-level",
                "ERROR",
            ],
            PROJECT_ROOT,
            agent_server_env,
        ),
        (
            "Web app",
            ("http://127.0.0.1:3000/", b"Sculpt Workflow Console"),
            [
                npm,
                "run",
                "start",
                "--",
                "--hostname",
                "127.0.0.1",
                "--port",
                "3000",
            ],
            WEB_ROOT,
            None,
        ),
    ]
    processes: list[subprocess.Popen[bytes]] = []
    previous_handlers = _install_terminal_exit_handlers()
    try:
        running_services = [
            f"{name} ({check[0]})"
            for name, check, _command, _cwd, _env in services
            if _url_is_ready(*check)
        ]
        if running_services:
            raise CliError(
                "Cannot bind all services to this terminal because these "
                "ViSculpt services are already running: "
                f"{', '.join(running_services)}. Stop their existing "
                "processes and run this command again."
            )
        for name, check, command, cwd, env in services:
            processes.append(_start_process(command, cwd=cwd, env=env))
        _wait_until_ready(
            {
                name: check
                for name, check, _command, _cwd, _env in services
            },
            processes,
        )
        if open_browser:
            webbrowser.open("http://127.0.0.1:3000/")
        print("ViSculpt is running. Press Ctrl+C to stop all services.")
        while True:
            for process in processes:
                return_code = process.poll()
                if return_code is not None:
                    raise CliError(
                        "A ViSculpt service stopped unexpectedly with code "
                        f"{return_code}."
                    )
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping ViSculpt services...")
    finally:
        for process in reversed(processes):
            _stop_process(process)
        _restore_signal_handlers(previous_handlers)


def _run_workflow(args: argparse.Namespace) -> None:
    try:
        workflow = SculptAgentWorkflow.from_config(args.config)
        state = workflow.invoke(args.instruction, run_id=args.run_id)
    except SculptWorkflowError as error:
        raise CliError(f"Workflow failed: {error}") from error
    summary = {
        "run_id": state["run_id"],
        "status": state["workflow_status"],
        "subtask_results": state["subtask_results"],
        "state_artifact_path": state["state_artifact_path"],
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visculpt",
        description="Visual-centric agentic geometry editing for Blender.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "setup",
        help="Install the main Python and Web app dependencies.",
    )
    subparsers.add_parser(
        "setup-sam3",
        help=(
            "Install SAM 3 with automatic Windows PyTorch backend selection."
        ),
    )
    addon_parser = subparsers.add_parser(
        "install-addon",
        help="Build and install the Blender RPC Add-on.",
    )
    addon_parser.add_argument(
        "--blender",
        type=Path,
        default=None,
        help="Optional path to the Blender executable.",
    )
    start_parser = subparsers.add_parser(
        "start",
        help="Start all local services and open the ViSculpt Web app.",
    )
    start_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the services without opening the default browser.",
    )
    workflow_parser = subparsers.add_parser(
        "sculpt-workflow",
        help="Run one workflow directly from the terminal.",
    )
    workflow_parser.add_argument(
        "instruction",
        help="Natural-language geometry editing instruction.",
    )
    workflow_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional centralized TOML configuration path.",
    )
    workflow_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable artifact run identifier.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one ViSculpt command and return its process exit code."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "setup":
            _setup_main_environment()
        elif args.command == "setup-sam3":
            _setup_sam3_environment()
        elif args.command == "install-addon":
            _install_addon(args.blender)
        elif args.command == "start":
            _start_services(open_browser=not args.no_browser)
        elif args.command == "sculpt-workflow":
            _run_workflow(args)
        else:
            raise CliError(f"Unsupported command: {args.command}")
    except CliError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
