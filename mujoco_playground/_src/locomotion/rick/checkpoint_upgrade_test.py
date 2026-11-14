"""Check that heading warm-start preserves outputs and normalization history."""

import unittest

from brax.training.acme import running_statistics
from brax.training import types
from jax import numpy as jp
import numpy as np

from mujoco_playground._src.locomotion.rick import checkpoint_upgrade


class HeadingUpgradeTest(unittest.TestCase):

  def test_policy_and_value_unchanged_for_arbitrary_heading_inputs(self):
    normalizer = running_statistics.RunningStatisticsState(
        count=types.UInt64(hi=1, lo=17), mean=jp.array([1., 2., 3.]),
        std=jp.array([.5, 2., 4.]), summed_variance=jp.array([2., 3., 4.]),
    )
    network = {'params': {'hidden_0': {
        'kernel': jp.arange(12, dtype=jp.float32).reshape(3, 4),
        'bias': jp.arange(4, dtype=jp.float32),
    }}}
    upgraded = checkpoint_upgrade.add_heading_inputs([normalizer, network, network], 3)
    obs = jp.array([.2, -.5, .7])
    for new_network in upgraded[1:]:
      old_layer = network['params']['hidden_0']
      layer = new_network['params']['hidden_0']
      expected = ((obs - normalizer.mean) / normalizer.std) @ old_layer['kernel'] + old_layer['bias']
      for heading in ([0., 1.], [1., 0.], [-.5, -.7]):
        inputs = jp.concatenate([obs, jp.array(heading)])
        actual = ((inputs - upgraded[0].mean) / upgraded[0].std) @ layer['kernel'] + layer['bias']
        np.testing.assert_array_equal(actual, expected)
    self.assertEqual(network['params']['hidden_0']['kernel'].shape, (3, 4))
    np.testing.assert_array_equal(upgraded[0].summed_variance[:3], normalizer.summed_variance)
    np.testing.assert_allclose(upgraded[0].summed_variance[3:], float(2**32 + 17))
    np.testing.assert_array_equal(upgraded[0].std[3:], [1., 1.])

  def test_unrelated_input_layout_is_rejected(self):
    normalizer = running_statistics.init_state(jp.zeros(4))
    with self.assertRaisesRegex(ValueError, 'flat Rick'):
      checkpoint_upgrade.add_heading_inputs([normalizer, {}, {}], 3)


class MotionUpgradeTest(unittest.TestCase):

  def parameters(self):
    count = float(2**32 + 17)
    std = jp.array([.5, 2., 1e-6, 4., 3.])
    normalizer = running_statistics.RunningStatisticsState(
        count=types.UInt64(hi=1, lo=17),
        mean=jp.array([1., 2., .05 - 2e-6, 3., 4.]),
        std=std, summed_variance=count * std**2,
    )
    network = {'params': {'hidden_0': {
        'kernel': jp.arange(20, dtype=jp.float32).reshape(5, 4) / 20,
        'bias': jp.arange(4, dtype=jp.float32),
    }}}
    return normalizer, network, network

  def first_layer(self, params, observation, network_index):
    normalized = running_statistics.normalize(jp.array(observation), params[0])
    layer = params[network_index]['params']['hidden_0']
    return normalized @ layer['kernel'] + layer['bias']

  def test_commands_initially_preserve_reference_walking_output(self):
    old = self.parameters()
    new = checkpoint_upgrade.add_motion_command_input(old, 2, .05)
    for i in (1, 2):
      expected = self.first_layer(old, [.2, -.5, .05, .7, .9], i)
      for speed, yaw in [(0., 0.), (.08, .6), (-.04, -.6)]:
        actual = self.first_layer(new, [.2, -.5, speed, yaw, .7, .9], i)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
      np.testing.assert_array_equal(new[i]['params']['hidden_0']['kernel'][2:4], 0.)
    self.assertEqual(old[0].mean.shape, (5,))
    self.assertEqual(old[1]['params']['hidden_0']['kernel'].shape, (5, 4))
    np.testing.assert_array_equal(new[0].mean[4:], old[0].mean[3:])
    np.testing.assert_allclose(new[0].summed_variance[2:4],
                               float(2**32+17) * np.array([.05**2, .3**2]), rtol=1e-6)
    updated = running_statistics.update(new[0], jp.zeros((32, 6)))
    np.testing.assert_allclose(updated.std[2:4], [.05, .3], rtol=1e-5)

  def test_turn_stage_preserves_zero_yaw_behavior_and_speed_response(self):
    params = checkpoint_upgrade.add_motion_command_input(self.parameters(), 2, .05)
    # Simulate straight-only training: yaw variance collapses, its row is arbitrary.
    normalizer = params[0].replace(std=params[0].std.at[3].set(1e-6))
    params = (normalizer, *params[1:])
    for network in params[1:]:
      layer = network['params']['hidden_0']
      layer['kernel'] = layer['kernel'].at[2].set(.5).at[3].set(5.)
    new = checkpoint_upgrade.activate_turn_command(params, 3)
    for i in (1, 2):
      for speed in (0., .03, .06):
        expected = self.first_layer(params, [.2, -.5, speed, 0., .7, .9], i)
        actual = self.first_layer(new, [.2, -.5, speed, .15, .7, .9], i)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
    self.assertAlmostEqual(float(new[0].std[3]), .3, places=6)
    self.assertGreater(float(running_statistics.update(new[0], jp.zeros((32, 6))).std[3]), .29)

  def test_incompatible_layout_and_statistics_rejected(self):
    params = self.parameters()
    with self.assertRaisesRegex(ValueError, 'layout'):
      checkpoint_upgrade.add_motion_command_input(params, 3, .05)
    with self.assertRaisesRegex(ValueError, 'positive scales'):
      checkpoint_upgrade.add_motion_command_input(params, 2, .05, speed_std=0.)
    with self.assertRaisesRegex(ValueError, 'Welford'):
      checkpoint_upgrade.add_motion_command_input(
          (params[0].replace(std_eps=.01), *params[1:]), 2, .05)

  def test_walking_config_compatibility(self):
    old = dict(heading_observation=True, command_history_length=12, ctrl_dt=.02,
               action_scale=.4, gyro_obs_scale=.25, gyro_bias_calibration=True,
               step_frequency=1.4)
    new = {**old, 'motion_command_version': 1}
    self.assertEqual(checkpoint_upgrade.validate_walking_upgrade(old, new, 107, 8), 104)
    for key in ('command_history_length', 'ctrl_dt', 'action_scale', 'gyro_obs_scale',
                'gyro_bias_calibration', 'step_frequency'):
      with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
        checkpoint_upgrade.validate_walking_upgrade(old, {**new, key: 99}, 107, 8)
    with self.assertRaisesRegex(ValueError, 'layout'):
      checkpoint_upgrade.validate_walking_upgrade(old, new, 106, 8)


if __name__ == '__main__':
  unittest.main()
