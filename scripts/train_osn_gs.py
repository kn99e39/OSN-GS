from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osn_gs import OSNGSFramework
from osn_gs.data import make_synthetic_scene


def main() -> None:
    scene = make_synthetic_scene()
    framework = OSNGSFramework()
    state = framework.train(scene)
    print(
        "OSN-GS smoke train complete: "
        f"iteration={state.iteration}, "
        f"loss={state.last_loss:.6f}, "
        f"certain={len(state.certain_gaussians)}, "
        f"uncertain={len(state.uncertain_gaussians)}"
    )


if __name__ == "__main__":
    main()
