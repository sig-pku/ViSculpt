"""Install the official SAM 3 checkpoint into the local service directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from uuid import uuid4

from sam3.model_builder import download_ckpt_from_cloud


DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent / "checkpoint" / "sam3.pt"
)


def ensure_sam3_checkpoint(
    destination: str | Path = DEFAULT_CHECKPOINT_PATH,
) -> Path:
    """Return a complete local checkpoint, downloading it when necessary."""
    target = Path(destination).expanduser().resolve()
    if target.is_file() and target.stat().st_size > 0:
        return target
    if target.exists() and not target.is_file():
        raise IsADirectoryError(f"Checkpoint destination is not a file: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    source = download_ckpt_from_cloud(
        local_dir=target.parent,
    ).expanduser().resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"Downloaded checkpoint is invalid: {source}")
    if source == target:
        return target

    temporary = target.with_name(f".{target.name}.{uuid4().hex}.part")
    try:
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != source.stat().st_size:
            raise OSError("The local checkpoint copy is incomplete.")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Local sam3.pt destination.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    existed = args.output.expanduser().is_file()
    checkpoint = ensure_sam3_checkpoint(args.output)
    action = "Using existing" if existed else "Installed"
    print(f"{action} SAM 3 checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_CHECKPOINT_PATH", "ensure_sam3_checkpoint", "main"]
