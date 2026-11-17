import subprocess
import sys
from pathlib import Path


def run_in_cli(
        args: tuple[str, ...],
        capture_output: bool=True,
        text: bool=True,
        check: bool=True
        ) -> subprocess.CompletedProcess[str]:
    """Run the provided arguments from command line and return the completed process"""
    
    result = subprocess.run(
        args,
        capture_output=capture_output,
        text=text,
        check=check,
    )

    return result


def run_pyright(path: Path) -> subprocess.CompletedProcess[str]:
    """Runs pyright on the specified file"""

    args: tuple[str, ...] = (
        sys.executable,
        "-m",
        "pyright",
        str(path),
    )

    res = run_in_cli(args)
    
    return res


def run_mypy(path: Path) -> subprocess.CompletedProcess[str]:
    """Runs mypy on the specified file"""

    args: tuple[str, ...] = (
        sys.executable,
        "-m",
        "mypy",
        str(path),
    )

    res = run_in_cli(
        args,
        check=False  # mypy process fails on errors, so don't check return code here
    )
    
    return res
