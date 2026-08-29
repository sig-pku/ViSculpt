from .cli import main as _main


def main() -> None:
    """Console-script adapter."""
    raise SystemExit(_main())
