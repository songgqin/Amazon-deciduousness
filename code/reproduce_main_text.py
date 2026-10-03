"""Run the repository-side main-text reproducibility gates.

The command renders the Figure 2 and Figure 3e checks and then runs the exact
local R4_2 Figure 4 GPP-SIF pixel-proportion calculation.  It exits non-zero
when either the Figure 3e proportions or the Figure 4 79.9% gate do not match
the Word main text.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from reproduce_main_figures import figure2, figure3_map
from verify_maintext_fig4 import run as verify_figure4


def main() -> None:
    failures = []
    print("Repository root:", ROOT)
    print(">>> Rendering Figure 2...")
    print("Wrote:", figure2())
    print(">>> Rendering and checking Figure 3e...")
    try:
        print("Wrote:", figure3_map(strict=True))
    except RuntimeError as error:
        failures.append(str(error))
        print("FAILED:", error)
    print(">>> Checking Figure 4 GPP-SIF pixel proportions...")
    try:
        verify_figure4()
    except (FileNotFoundError, RuntimeError) as error:
        failures.append(str(error))
        print("FAILED:", error)
    if failures:
        raise RuntimeError("Main-text reproducibility gates failed:\n" + "\n".join(failures))
    print("Main-text reproducibility gates passed.")


if __name__ == "__main__":
    main()
