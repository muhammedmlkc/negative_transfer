from __future__ import annotations

import subprocess
import sys


COMMANDS = [
    [
        sys.executable,
        "safe_tcn_lab/run_experiment.py",
        "--dataset",
        "gefcom",
        "--smoke",
        "--methods",
        "persistence",
        "tcn_local",
        "safe_tcn",
        "--max_sources",
        "2",
    ],
    [
        sys.executable,
        "safe_tcn_lab/run_experiment.py",
        "--dataset",
        "sdwpf",
        "--smoke",
        "--methods",
        "persistence",
        "tcn_local",
        "--max_sources",
        "2",
    ],
]


def main() -> None:
    for command in COMMANDS:
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
