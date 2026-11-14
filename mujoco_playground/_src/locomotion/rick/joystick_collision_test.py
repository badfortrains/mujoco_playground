"""Regression tests for Rick's four-sphere foot collisions."""

import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from etils import epath
import jax
from jax import numpy as jp
import mujoco
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.rick import joystick
from mujoco_playground._src.locomotion.rick import rick_constants


class CornerSphereCollisionTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # A self-contained model exercises the XML schema without fetching assets.
    feet, pairs, sensors, actuators = [], [], [], []
    for side, x in (('left', .06), ('right', -.06)):
      spheres = []
      for corner, (sx, sy) in enumerate(
          ((-.0128, -.0375), (.0128, -.0375),
           (-.0128, .0375), (.0128, .0375)), start=1):
        name = f'{side}_foot_collision_{corner}'
        spheres.append(f'<geom name="{name}" type="sphere" '
                       f'pos="{sx} {sy} -.015" size=".005" density="0" '
                       'contype="1" conaffinity="1"/>')
        pairs.append(f'<pair name="{side}_foot_floor_{corner}" '
                     f'geom1="{name}" geom2="floor" friction=".35 .35"/>')
      feet.append(f'<body name="{side}" pos="{x} 0 -.08">'
                  f'<joint name="{side}" axis="1 0 0"/>'
                  '<geom type="sphere" size=".01" mass=".03"/>'
                  + ''.join(spheres)
                  + f'<site name="{side}_foot_center" pos="0 0 -.02"/>'
                  '</body>')
      sensors.append(f'<contact name="{side}_foot_floor_found" '
                     f'body1="{side}" geom2="floor" '
                     'reduce="mindist" num="1" data="found"/>')
      actuators.append(f'<position joint="{side}" kp="5" kv=".02" '
                       'ctrlrange="-1 1" forcerange="-.2 .2"/>')
    xml = ('<mujoco><default><geom contype="0" conaffinity="0"/>'
           '<joint damping=".002" armature=".0005"/></default><worldbody>'
           '<geom name="floor" type="plane" size="1 1 .05" '
           'contype="1" conaffinity="1"/>'
           '<body name="body"><freejoint/>'
           '<geom type="sphere" size=".02" mass=".3"/>'
           '<site name="imu"/>' + ''.join(feet) + '</body></worldbody>'
           '<contact>' + ''.join(pairs) + '</contact>'
           '<sensor>' + ''.join(sensors) + '</sensor>'
           '<actuator>' + ''.join(actuators) + '</actuator>'
           '<keyframe><key name="basic" qpos="0 0 .1 1 0 0 0 0 0" '
           'ctrl="0 0"/></keyframe></mujoco>')
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    path = epath.Path(directory.name) / 'spheres.xml'
    path.write_text(xml)
    with mock.patch.object(rick_constants, 'task_to_xml', return_value=path):
      self.env = joystick.Joystick()

  def test_friction_randomization_reaches_all_eight_floor_pairs(self):
    env = self.env
    self.assertEqual(env._foot_sphere_geom_ids.shape, (2, 4))
    np.testing.assert_array_equal(
        np.sort(env._foot_floor_pair_ids), np.arange(8))
    state = env.reset(jax.random.PRNGKey(0))
    state.info['foot_friction'] = jp.array(.23)
    # Inspect the model actually sent to physics, keeping this test lightweight.
    with mock.patch.object(mjx_env, 'step', side_effect=lambda m, d, *args: d) as step:
      result = env.step(state, jp.zeros(env.action_size))
    model = step.call_args.args[0]
    np.testing.assert_allclose(model.pair_friction[:, :2], .23)
    np.testing.assert_array_equal(
        model.pair_friction[:, 2:], env.mjx_model.pair_friction[:, 2:])
    self.assertTrue(np.isfinite(result.reward))

  def test_tilted_clearances_match_native_sphere_plane_distances(self):
    env = self.env
    model = env.mj_model
    data = mujoco.MjData(model)
    clearance = jax.jit(lambda positions: env._get_foot_clearances(
        SimpleNamespace(geom_xpos=positions)))
    for axis, angle in (((1, 0, 0), 0), ((1, 0, 0), .4),
                        ((0, 1, 0), -.35), ((1, 1, 0), .5)):
      mujoco.mj_resetDataKeyframe(model, data, model.key('basic').id)
      axis = np.array(axis, dtype=float)
      mujoco.mju_axisAngle2Quat(data.qpos[3:7], axis / np.linalg.norm(axis), angle)
      mujoco.mj_forward(model, data)
      expected = np.array([
          [mujoco.mj_geomDistance(model, data, int(geom), model.geom('floor').id,
                                 1.0, None) for geom in foot]
          for foot in np.asarray(env._foot_sphere_geom_ids)
      ])
      actual = clearance(jp.array(data.geom_xpos))
      np.testing.assert_allclose(actual, [expected[:, :2].min(axis=1),
                                         expected[:, 2:].min(axis=1)], atol=1e-8)

  def test_each_corner_can_report_foot_contact(self):
    env = self.env
    data = mujoco.MjData(env.mj_model)
    self.assertEqual(env._foot_contact_sensor_adrs.shape, (2, 4))
    for side in range(2):
      for corner in range(4):
        data.qpos[:] = env._default_qpos
        # Lower only this sphere, leaving every other sphere above the floor.
        model = env.mj_model
        positions = model.geom_pos.copy()
        try:
          model.geom_pos[np.asarray(env._foot_sphere_geom_ids), 2] += .05
          model.geom_pos[int(env._foot_sphere_geom_ids[side, corner]), 2] -= .051
          mujoco.mj_forward(model, data)
          contacts = np.asarray(data.sensordata)[np.asarray(env._foot_contact_sensor_adrs)] > 0
          expected = np.zeros((2, 4), dtype=bool)
          expected[side, corner] = True
          np.testing.assert_array_equal(contacts, expected)
        finally:
          model.geom_pos[:] = positions


if __name__ == '__main__':
  unittest.main()
