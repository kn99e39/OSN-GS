from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UpdateSchedule:
    curve_update_interval: int = 100
    density_control_interval: int = 100

    def should_update_curves(self, iteration: int) -> bool:
        return self.curve_update_interval > 0 and iteration > 0 and iteration % self.curve_update_interval == 0

    def should_run_density_control(self, iteration: int) -> bool:
        return (
            self.density_control_interval > 0
            and iteration > 0
            and iteration % self.density_control_interval == 0
        )
