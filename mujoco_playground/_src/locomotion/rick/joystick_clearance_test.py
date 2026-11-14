"""Tests for Rick's phase-aware foot-clearance costs."""

from absl.testing import absltest
import jax
from jax import numpy as jp
import numpy as np

from mujoco_playground._src.locomotion.rick import joystick


class FootClearanceCostTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = object.__new__(joystick.Joystick)
    self.env._natural_clearance_scale = 0.012
    self.env._swing_height_undershoot_cost_weight = 0.5
    self.env._swing_height_overshoot_cost_weight = 0.25
    self.env._swing_height_overshoot_tolerance = 0.25
    self.env._excessive_swing_height_cost_weight = 1.0
    self.env._excessive_swing_height_threshold = 1.25
    self.costs = jax.jit(self.env._get_foot_height_costs)

  def test_default_config_enables_clearance_costs(self):
    config = joystick.default_config()
    self.assertEqual(config.swing_height_undershoot_cost_weight, 0.5)
    self.assertEqual(config.swing_height_overshoot_cost_weight, 0.25)
    self.assertEqual(config.swing_height_overshoot_tolerance, 0.25)
    self.assertEqual(config.excessive_swing_height_cost_weight, 1.0)
    self.assertEqual(config.excessive_swing_height_threshold, 1.25)

  def test_costs_raise_undershoot_and_limit_only_large_overshoot(self):
    c0 = self.env._natural_clearance_scale
    phase = jp.array(jp.pi / 2)

    tracking, excessive = self.costs(
        phase, jp.array([0.0, 3.0 * c0]), jp.array([0.0, 3.0 * c0])
    )
    np.testing.assert_allclose(tracking, [0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(excessive, [0.0, 0.0], atol=1e-6)

    for clearance in (c0, 1.25 * c0):
      tracking, excessive = self.costs(
          phase,
          jp.array([clearance, 3.0 * c0]),
          jp.array([clearance, 3.0 * c0]),
      )
      np.testing.assert_allclose(tracking, [0.0, 0.0], atol=1e-6)
      np.testing.assert_allclose(excessive, [0.0, 0.0], atol=1e-6)

    tracking, excessive = self.costs(
        phase, jp.array([2.0 * c0, 0.0]), jp.array([2.0 * c0, 0.0])
    )
    np.testing.assert_allclose(tracking, [0.140625, 0.0], atol=1e-6)
    np.testing.assert_allclose(excessive, [0.5625, 0.0], atol=1e-6)

  def test_costs_are_symmetric_across_opposite_half_cycles(self):
    c0 = self.env._natural_clearance_scale
    left = self.costs(
        jp.array(jp.pi / 2), jp.zeros(2), jp.zeros(2)
    )
    right = self.costs(
        jp.array(-jp.pi / 2), jp.zeros(2), jp.zeros(2)
    )
    np.testing.assert_allclose(left[0], right[0][::-1], atol=1e-6)
    np.testing.assert_allclose(left[1], right[1][::-1], atol=1e-6)

    # The lower sole corner controls clearance, so lifting only the toe still
    # incurs the full undershoot cost.
    tracking, excessive = self.costs(
        jp.array(jp.pi / 2), jp.array([c0, 0.0]), jp.zeros(2)
    )
    np.testing.assert_allclose(tracking, [0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(excessive, [0.0, 0.0], atol=1e-6)


if __name__ == '__main__':
  absltest.main()
