"""Regression tests for Rick's mesh foot collisions."""

import tempfile
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


class MeshCollisionTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # A self-contained model exercises the XML schema without fetching assets.
    vertices = (
        np.indices((2, 2, 2)).reshape(3, -1).T * 2 - 1
    ) * [.0128, .0375, .005]
    vertex_text = ' '.join(str(value) for value in vertices.flat)
    feet, pairs, sensors, actuators = [], [], [], []
    for side, x in (('left', .06), ('right', -.06)):
      name = f'{side}_foot_collision'
      feet.append(
          f'<body name="{side}" pos="{x} 0 -.08">'
          f'<joint name="{side}" axis="1 0 0"/>'
          f'<geom name="{name}" type="mesh" mesh="sole" '
          'pos="0 0 -.015" mass=".03" contype="1" conaffinity="1"/>'
          f'<site name="{side}_foot_center" pos="0 0 -.02"/>'
          '</body>'
      )
      pairs.append(
          f'<pair name="{side}_foot_floor" geom1="{name}" '
          'geom2="floor" friction=".35 .35"/>'
      )
      sensors.append(
          f'<contact name="{side}_foot_floor_found" geom1="{name}" '
          'geom2="floor" reduce="mindist" num="1" data="found"/>'
      )
      actuators.append(
          f'<position joint="{side}" kp="5" kv=".02" '
          'ctrlrange="-1 1" forcerange="-.2 .2"/>'
      )
    xml = (
        f'<mujoco><asset><mesh name="sole" vertex="{vertex_text}"/></asset>'
        '<default><geom contype="0" conaffinity="0"/>'
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
        'ctrl="0 0"/></keyframe></mujoco>'
    )
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    path = epath.Path(directory.name) / 'mesh_feet.xml'
    path.write_text(xml)
    with mock.patch.object(rick_constants, 'task_to_xml', return_value=path):
      self.env = joystick.Joystick()

  def test_friction_randomization_reaches_both_mesh_floor_pairs(self):
    env = self.env
    for side in ('left', 'right'):
      self.assertEqual(
          env.mj_model.geom(f'{side}_foot_collision').type[0],
          mujoco.mjtGeom.mjGEOM_MESH,
      )
    np.testing.assert_array_equal(
        np.sort(env._foot_floor_pair_ids), np.arange(2))
    self.assertEqual(env._foot_contact_sensor_adrs.shape, (2, 1))
    state = env.reset(jax.random.PRNGKey(0))
    state.info['foot_friction'] = jp.array(.23)
    # Inspect the model actually sent to physics, keeping this test lightweight.
    with mock.patch.object(
        mjx_env, 'step', side_effect=lambda model, data, *args: data
    ) as step:
      result = env.step(state, jp.zeros(env.action_size))
    model = step.call_args.args[0]
    np.testing.assert_allclose(model.pair_friction[:, :2], .23)
    np.testing.assert_array_equal(
        model.pair_friction[:, 2:], env.mjx_model.pair_friction[:, 2:])
    self.assertTrue(np.isfinite(result.reward))

  def test_both_meshes_report_foot_contact(self):
    env = self.env
    model = env.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = env._default_qpos
    data.qpos[2] -= .01
    mujoco.mj_forward(model, data)
    contacts = (
        np.asarray(data.sensordata)[
            np.asarray(env._foot_contact_sensor_adrs)
        ] > 0
    )
    np.testing.assert_array_equal(contacts, np.ones((2, 1), dtype=bool))


if __name__ == '__main__':
  unittest.main()
