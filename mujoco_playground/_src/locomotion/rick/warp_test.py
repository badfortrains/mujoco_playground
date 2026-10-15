"""Warp state-selection regression tests and a real Rick backend smoke test.

State-selection tests use actual Warp Data on CPU, without executing physics.
The real physics test runs JAX on CPU and Warp when a CUDA GPU is available.
Rick assets come from the neighboring rick_v2 checkout; tests never fetch them.
"""

from pathlib import Path
import unittest
from unittest import mock

from etils import epath
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import rollout_diagnostics
from mujoco_playground._src import wrapper
from mujoco_playground._src.locomotion.rick import joystick
from mujoco_playground._src.locomotion.rick import rick_constants


class WarpCounter:
  """Exercise real Warp buffers and mixed episode endings without CUDA."""

  dt = 0.02
  action_size = 1

  def __init__(self):
    from mujoco.mjx._src import io as mjx_io
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <worldbody><body><joint name="slide" type="slide"/>
        <geom size=".1"/></body></worldbody>
      <actuator><motor joint="slide"/></actuator></mujoco>''')
    # Allocate the real Warp layout on CPU, without invoking GPU-only physics.
    self.data = mjx_io._make_data_warp(
        model, device=jax.devices('cpu')[0], naconmax=2, njmax=4
    )

  def reset(self, key):
    return mjx_env.State(
        self.data, jp.zeros(1), jp.array(0.0), jp.array(0.0),
        {'count': jp.array(0.0)}, {'sample': jax.random.uniform(key)},
    )

  def step(self, state, action):
    # Stepping a finished world produces NaN, exposing any missing freeze.
    qpos = jp.where(state.done, jp.nan, state.data.qpos + action)
    data = state.data.replace(qpos=qpos, time=state.data.time + self.dt)
    data = data.tree_replace({
        # Deliberately the same leading size as the batch, but NOT per world.
        '_impl.contact__dist': jp.full_like(data._impl.contact__dist, 7.0),
    })
    return state.replace(
        data=data, obs=qpos, reward=qpos[0],
        done=(qpos[0] >= 2).astype(jp.float32), metrics={'count': qpos[0]},
    )


@unittest.skipUnless(hasattr(mjx.Data, 'where'), 'Requires current MJX-Warp')
class WarpSelectionTest(unittest.TestCase):

  def test_autoreset_preserves_shared_contact_buffers(self):
    for full_reset in (False, True):
      with self.subTest(full_reset=full_reset):
        env = wrapper.wrap_for_brax_training(WarpCounter(), full_reset=full_reset)
        state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
        state = jax.jit(env.step)(state, jp.array([[2.0], [1.0]]))
        np.testing.assert_array_equal(state.done, [1, 0])
        np.testing.assert_array_equal(state.data.qpos[:, 0], [0, 1])
        # Contacts must remain the current batch-wide buffers even for world 0.
        np.testing.assert_array_equal(state.data._impl.contact__dist, [7, 7])

  def test_diagnostics_freeze_each_world_at_its_own_terminal(self):
    def policy(obs, key):
      del obs
      # Seeds are deterministic; different actions exercise mixed endings.
      return jp.where(jax.random.bernoulli(key), jp.array([1.0]), jp.array([2.0])), {}

    for repeat in (1, 4):
      with self.subTest(action_repeat=repeat):
        initial, trace = rollout_diagnostics.collect(WarpCounter(), policy, [0, 1], 6, repeat)
        again, trace_again = rollout_diagnostics.collect(WarpCounter(), policy, [0, 1], 6, repeat)
        np.testing.assert_array_equal(trace['data']['qpos'], trace_again['data']['qpos'])
        np.testing.assert_array_equal(initial['obs'], again['obs'])
        for world in range(2):
          terminal = np.flatnonzero(trace['done'][world])[0]
          self.assertTrue(trace['valid'][world, terminal])
          self.assertFalse(trace['valid'][world, terminal + 1:].any())
          tail = trace['data']['qpos'][world, terminal:]
          np.testing.assert_array_equal(tail, np.broadcast_to(tail[0], tail.shape))
          self.assertTrue(np.isfinite(tail).all())
          np.testing.assert_array_equal(trace['reward'][world, terminal + 1:], 0)


class RickPhysicsTest(unittest.TestCase):

  def test_jax_batched_randomized_physics_and_diagnostics(self):
    self._check_backend('jax')

  def test_warp_batched_randomized_physics_and_diagnostics(self):
    if not any(d.platform == 'gpu' for d in jax.devices()):
      self.skipTest('MJX-Warp physics requires a CUDA GPU')
    self._check_backend('warp')

  def _check_backend(self, backend):
    xml = Path(__file__).resolve().parents[5] / 'rick_v2/v4Robot_v81.xml'
    if not xml.is_file():
      self.skipTest('Neighboring rick_v2 assets unavailable')
    config = joystick.default_config()
    config.impl = backend
    config.naconmax = 16 * 2
    with mock.patch.object(rick_constants, 'task_to_xml', return_value=epath.Path(xml)):
      env = joystick.Joystick(config=config)
    self.assertEqual(env.mjx_model.impl.value, backend)
    keys = jax.random.split(jax.random.PRNGKey(4), 2)
    state = jax.jit(jax.vmap(env.reset))(keys)
    self.assertFalse(np.array_equal(state.info['servo_strength'][0], state.info['servo_strength'][1]))
    state = jax.jit(jax.vmap(env.step))(state, jp.zeros((2, env.action_size)))
    self.assertEqual(state.obs.shape, (2, 105))
    self.assertTrue(np.isfinite(state.data.qpos).all())
    policy = lambda obs, key: (jp.zeros(env.action_size), {})
    _, trace = rollout_diagnostics.collect(env, policy, [0, 1], 2)
    self.assertTrue(np.isfinite(trace['data']['qpos']).all())


if __name__ == '__main__':
  unittest.main()
