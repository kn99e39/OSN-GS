from __future__ import annotations

"""Numpy prototype용 high-level framework.

이 파일은 초기 smoke test와 알고리즘 스케치를 위한 가벼운 API다.
실제 CUDA/학습 결과 생성 경로는 `core/torch_trainer.py`와
`scripts/train_osn_gs_torch.py`를 사용한다.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.core.pipeline import OSNGSPipeline, PipelineConfig
from osn_gs.core.state import OSNGSState
from osn_gs.core.trainer import OSNGSTrainer, TrainingConfig
from osn_gs.data.scene_loader import Scene
from osn_gs.optim.schedulers import UpdateSchedule


@dataclass
class OSNGSConfig:
    """Numpy prototype pipeline/training 설정 묶음."""

    pipeline: PipelineConfig
    training: TrainingConfig

    @classmethod
    def default(cls) -> "OSNGSConfig":
        return cls(pipeline=PipelineConfig(), training=TrainingConfig())

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "OSNGSConfig":
        pipeline = PipelineConfig(**config.get("pipeline", {}))
        schedule = UpdateSchedule(**config.get("schedule", {}))
        training_data = dict(config.get("training", {}))
        training_data["schedule"] = schedule
        training = TrainingConfig(**training_data)
        return cls(pipeline=pipeline, training=training)


class OSNGSFramework:
    """Numpy prototype을 한 번에 실행하기 위한 작은 facade."""

    def __init__(self, config: OSNGSConfig | None = None) -> None:
        self.config = config or OSNGSConfig.default()
        self.pipeline = OSNGSPipeline(self.config.pipeline)
        self.trainer = OSNGSTrainer(self.pipeline, config=self.config.training)

    def initialize(self, scene: Scene) -> OSNGSState:
        # scene의 initial_gaussians를 복사해 prototype pipeline state를 만든다.
        return self.pipeline.initialize(scene.initial_gaussians.clone())

    def train(self, scene: Scene) -> OSNGSState:
        # 실제 gradient 학습이 아니라 numpy 기반 smoke train loop다.
        return self.trainer.train(scene)
