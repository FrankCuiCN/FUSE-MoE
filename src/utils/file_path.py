import argparse
from pathlib import Path


def file_path(string: str) -> Path:
    """Validate that the argument exists *and* is a file."""
    p = Path(string)
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"'{p}' is not a file")
    return p
