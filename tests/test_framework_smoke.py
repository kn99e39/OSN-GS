import unittest

from osn_gs import OSNGSFramework
from osn_gs.core.framework import OSNGSConfig
from osn_gs.data import make_synthetic_scene


class FrameworkSmokeTest(unittest.TestCase):
    def test_initialize_builds_surface_and_uncertain_gaussians(self):
        scene = make_synthetic_scene(point_count=16, image_size=8)
        framework = OSNGSFramework()

        state = framework.initialize(scene)

        self.assertIsNotNone(state.nurbs_surface)
        self.assertGreater(len(state.base_curves), 0)
        self.assertGreater(len(state.occlusion_curves), 0)
        self.assertGreater(len(state.uncertain_gaussians), 0)

    def test_train_runs_for_requested_iterations(self):
        config = OSNGSConfig.from_dict({"training": {"iterations": 3}})
        scene = make_synthetic_scene(point_count=16, image_size=8)
        framework = OSNGSFramework(config)

        state = framework.train(scene)

        self.assertEqual(state.iteration, 3)
        self.assertGreaterEqual(state.last_loss, 0.0)


if __name__ == "__main__":
    unittest.main()
