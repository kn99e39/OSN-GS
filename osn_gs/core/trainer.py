from __future__ import annotations

from dataclasses import dataclass, field

from osn_gs.core.pipeline import OSNGSPipeline
from osn_gs.core.state import OSNGSState
from osn_gs.data.scene_loader import Scene
from osn_gs.gaussian.density_control import update_uncertain_confidence_from_residual
from osn_gs.losses.image_similarity import mse_loss
from osn_gs.losses.nurbs_regularization import surface_smoothness_loss
from osn_gs.losses.uncertainty import residual_weighted_uncertainty_loss
from osn_gs.optim.curve_update import decide_curve_update
from osn_gs.optim.schedulers import UpdateSchedule
from osn_gs.render.prototype_renderer import OSNPrototypeRenderer


@dataclass
class TrainingConfig:
    iterations: int = 10
    batch_size: int = 1
    surface_loss_weight: float = 0.01
    uncertainty_loss_weight: float = 0.1
    schedule: UpdateSchedule = field(default_factory=UpdateSchedule)


class OSNGSTrainer:
    def __init__(
        self,
        pipeline: OSNGSPipeline,
        renderer: OSNPrototypeRenderer | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.renderer = renderer or OSNPrototypeRenderer()
        self.config = config or TrainingConfig()

    def train(self, scene: Scene) -> OSNGSState:
        state = self.pipeline.initialize(scene.initial_gaussians.clone())
        for iteration in range(1, self.config.iterations + 1):
            batch = scene.sample_views(self.config.batch_size)
            rendered = self.renderer.render(
                state.certain_gaussians,
                state.uncertain_gaussians,
                batch.cameras,
            )
            image_loss = mse_loss(rendered, batch.images)
            uncertainty_loss = residual_weighted_uncertainty_loss(image_loss, state.uncertain_gaussians)
            surface_loss = surface_smoothness_loss(state.nurbs_surface) if state.nurbs_surface else 0.0
            total_loss = (
                image_loss
                + self.config.uncertainty_loss_weight * uncertainty_loss
                + self.config.surface_loss_weight * surface_loss
            )
            update_uncertain_confidence_from_residual(state.uncertain_gaussians, image_loss)
            state.iteration = iteration
            state.last_loss = total_loss
            decision = decide_curve_update(image_loss, state.uncertain_gaussians)
            if self.config.schedule.should_update_curves(iteration) or decision.should_rebuild_surface:
                self.pipeline.rebuild_surface(state)
        return state

