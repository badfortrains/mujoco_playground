"""Regression tests for recorded action/state alignment and artifact playback."""
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

from flax import struct
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from mujoco_playground._src import rollout_diagnostics as diagnostics


@struct.dataclass
class Data:
  qpos: object
  qvel: object
  time: object
  ctrl: object
  actuator_force: object
  sensordata: object
  site_xpos: object
  mocap_pos: object
  mocap_quat: object
  xfrc_applied: object


@struct.dataclass
class State:
  data: object
  obs: object
  info: object
  metrics: object
  reward: object
  done: object


class CounterEnv:
  dt = 0.02

  def reset(self, key):
    zero = jp.array(0.0)
    data = Data(jp.zeros(1), jp.zeros(1), zero, jp.zeros(1), jp.zeros(1),
                jp.zeros(1), jp.zeros((1, 3)), jp.zeros((0, 3)),
                jp.zeros((0, 4)), jp.zeros((2, 6)))
    return State(data, jp.zeros(1), {'sample': jax.random.uniform(key)},
                 {'reward_alive': zero}, zero, zero)

  def step(self, state, action):
    # NaNs would expose accidental stepping beyond terminal.
    q = jp.where(state.done, jp.nan, state.data.qpos + 1)
    return state.replace(
        data=state.data.replace(qpos=q, time=state.data.time + self.dt, ctrl=action),
        obs=q, reward=q[0], done=(q[0] >= 2).astype(jp.float32),
        metrics={'reward_alive': q[0]},
    )


def policy(obs, key):
  del key
  return obs + 10, {}


class DiagnosticsTest(unittest.TestCase):
  def test_terminal_transition_and_action_alignment(self):
    initial, trace = diagnostics.collect(CounterEnv(), policy, [0, 1], 5)
    np.testing.assert_array_equal(trace['valid'], [[1, 1, 0, 0, 0]] * 2)
    np.testing.assert_allclose(trace['data']['qpos'][0, :, 0], [1, 2, 2, 2, 2])
    np.testing.assert_allclose(trace['obs'][0, :2, 0], [0, 1])
    np.testing.assert_allclose(trace['next_obs'][0, :2, 0], [1, 2])
    np.testing.assert_allclose(trace['action'][0, :, 0], [10, 11, 0, 0, 0])
    np.testing.assert_allclose(trace['reward'][0], [1, 2, 0, 0, 0])
    np.testing.assert_allclose(initial['data']['qpos'], [[0], [0]])
    summary = diagnostics.summarize(trace, [0, 1])
    self.assertEqual(summary[0]['episode_return'], 3)
    self.assertEqual(summary[0]['valid_steps'], 2)
    self.assertTrue(summary[0]['terminated'])

  def test_repeat_stops_on_terminal_and_counts_reward_once(self):
    _, trace = diagnostics.collect(CounterEnv(), policy, [0], 3, action_repeat=4)
    np.testing.assert_array_equal(trace['valid'], [[1, 0, 0]])
    np.testing.assert_allclose(trace['reward'], [[3, 0, 0]])
    np.testing.assert_allclose(trace['data']['time'], [[0.04, 0.04, 0.04]])

  def test_seed_reproducibility_and_diversity(self):
    first, trace = diagnostics.collect(CounterEnv(), policy, [3, 9], 2)
    again, trace_again = diagnostics.collect(CounterEnv(), policy, [3, 9], 2)
    np.testing.assert_array_equal(first['info']['sample'], again['info']['sample'])
    self.assertNotEqual(first['info']['sample'][0], first['info']['sample'][1])
    np.testing.assert_array_equal(trace['action'], trace_again['action'])

  def test_bundle_round_trip_and_rscope_assets(self):
    env = CounterEnv()
    initial, trace = diagnostics.collect(env, policy, [0, 1], 4)
    with tempfile.TemporaryDirectory() as root:
      root = Path(root)
      xml = '<mujoco><worldbody><body><joint type="slide"/><geom size=".1"/></body></worldbody></mujoco>'
      env.xml_path = str(root / 'model.xml')
      Path(env.xml_path).write_text(xml)
      env.model_assets = {}
      env.mj_model = mujoco.MjModel.from_xml_string(xml)
      out = root / 'bundle'
      metadata = diagnostics.write_bundle(out, env, initial, trace, [0, 1], {'checkpoint': '123'})
      with np.load(out / 'traces.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['trace/valid'], trace['valid'])
        np.testing.assert_array_equal(arrays['initial/info/sample'], initial['info']['sample'])
      self.assertEqual(metadata['checkpoint'], '123')
      self.assertIn('traces.npz', metadata['sha256'])
      self.assertTrue((out / 'seed_0.png').is_file())
      loaded_model = mujoco.MjModel.from_binary_path(str(out / 'model.mjb'))
      self.assertEqual(loaded_model.nq, env.mj_model.nq)
      with (out / 'rscope/rollout.mj_unroll').open('rb') as f:
        rollout = pickle.load(f)
      self.assertEqual(rollout.qpos.shape, (4, 2, 1))
      np.testing.assert_array_equal(rollout.metrics['valid'], trace['valid'].T)
      with (out / 'rscope/rscope_meta.pkl').open('rb') as f:
        meta = pickle.load(f)
      model = mujoco.MjModel.from_xml_string(
          meta['model_assets'][meta['xml_path']].decode(), assets=meta['model_assets']
      )
      self.assertEqual(model.nq, env.mj_model.nq)

  def test_video_uses_the_saved_states_and_excludes_padding(self):
    env = CounterEnv()
    _, trace = diagnostics.collect(env, policy, [0], 5)
    env.render = mock.Mock(return_value=[np.zeros((4, 4, 3), np.uint8)])
    with mock.patch('mediapy.write_video') as write:
      diagnostics.render_videos(env, trace, '/tmp/unused', 1)
    states = env.render.call_args.args[0]
    self.assertEqual(len(states), 1)  # Two valid steps, render every other step.
    np.testing.assert_array_equal(states[0].data.qpos, trace['data']['qpos'][0, 0])
    self.assertEqual(write.call_args.kwargs['fps'], 25)


if __name__ == '__main__':
  unittest.main()
