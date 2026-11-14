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


if __name__ == '__main__':
  unittest.main()
