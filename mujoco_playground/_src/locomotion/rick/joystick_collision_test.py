"""Regression tests for Rick's mesh foot collisions."""

import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

from etils import epath
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
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


class CapsuleCollisionTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    feet, pairs, sensors, actuators = [], [], [], []
    for side, body_x in (('left', .06), ('right', -.06)):
      geoms = []
      for index, capsule_x in enumerate((-.006, .006), start=1):
        name = f'{side}_foot_collision_{index}'
        geoms.append(
            f'<geom name="{name}" type="capsule" '
            f'fromto="{capsule_x} -.03 -.01 {capsule_x} .03 -.01" '
            'size=".01" mass=".015" contype="1" conaffinity="1"/>'
        )
        pairs.append(
            f'<pair name="{side}_foot_floor_{index}" geom1="{name}" '
            'geom2="floor" friction=".35 .35"/>'
        )
      feet.append(
          f'<body name="{side}" pos="{body_x} 0 -.08">'
          f'<joint name="{side}" axis="1 0 0"/>'
          + ''.join(geoms)
          + f'<site name="{side}_foot_center" pos="0 0 -.02"/>'
          '</body>'
      )
      sensors.append(
          f'<contact name="{side}_foot_floor_found" body1="{side}" '
          'geom2="floor" reduce="mindist" num="1" data="found"/>'
      )
      actuators.append(
          f'<position joint="{side}" kp="5" kv=".02" '
          'ctrlrange="-1 1" forcerange="-.2 .2"/>'
      )
    xml = (
        '<mujoco><default><geom contype="0" conaffinity="0"/>'
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
    path = epath.Path(directory.name) / 'capsule_feet.xml'
    path.write_text(xml)
    self._xml_path = path
    with mock.patch.object(rick_constants, 'task_to_xml', return_value=path):
      self.env = joystick.Joystick()

  def test_indexed_capsules_expand_pairs_and_sensors(self):
    env = self.env
    self.assertEqual(np.asarray(env._foot_capsule_geom_ids).shape, (2, 2))
    self.assertIsNone(env._foot_sphere_geom_ids)
    for side in ('left', 'right'):
      for index in (1, 2):
        self.assertEqual(
            env.mj_model.geom(f'{side}_foot_collision_{index}').type[0],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
        )
    np.testing.assert_array_equal(
        np.sort(env._foot_floor_pair_ids), np.arange(4))
    self.assertEqual(env._foot_contact_sensor_adrs.shape, (2, 2))

  def test_capsule_endpoint_clearance_tracks_pitch(self):
    env = self.env
    data = mujoco.MjData(env.mj_model)
    data.qpos[:] = env._default_qpos
    mujoco.mj_forward(env.mj_model, data)
    toe, heel = env._get_foot_clearances(mjx.put_data(env.mj_model, data))
    np.testing.assert_allclose(toe, 0.0, atol=1e-6)
    np.testing.assert_allclose(heel, 0.0, atol=1e-6)

    data.qpos[7] = 0.1
    mujoco.mj_forward(env.mj_model, data)
    toe, heel = env._get_foot_clearances(mjx.put_data(env.mj_model, data))
    self.assertLess(float(toe[0]), -0.002)
    self.assertGreater(float(heel[0]), 0.002)
    np.testing.assert_allclose(toe[1], 0.0, atol=1e-6)
    np.testing.assert_allclose(heel[1], 0.0, atol=1e-6)

  def test_front_caps_join_contact_friction_and_clearance(self):
    tree = ET.parse(self._xml_path.as_posix())
    root = tree.getroot()
    for side in ('left', 'right'):
      body = root.find(f".//body[@name='{side}']")
      for index, x in enumerate((-.006, .006), start=1):
        name = f'{side}_foot_front_{index}'
        ET.SubElement(body, 'geom', {
            'name': name, 'type': 'sphere', 'pos': f'{x} -.03 -.006',
            'size': '.014', 'density': '0', 'contype': '1',
            'conaffinity': '1',
        })
        ET.SubElement(root.find('contact'), 'pair', {
            'name': f'{side}_foot_front_floor_{index}',
            'geom1': name, 'geom2': 'floor', 'friction': '.35 .35',
        })
    front_path = self._xml_path.parent / 'front_cap_feet.xml'
    tree.write(front_path.as_posix())
    with mock.patch.object(rick_constants, 'task_to_xml', return_value=front_path):
      env = joystick.Joystick()

    self.assertEqual(np.asarray(env._foot_front_geom_ids).shape, (2, 2))
    np.testing.assert_array_equal(np.sort(env._foot_floor_pair_ids), np.arange(8))
    self.assertEqual(env._foot_contact_sensor_adrs.shape, (2, 4))
    data = mujoco.MjData(env.mj_model)
    data.qpos[:] = env._default_qpos
    data.qpos[2] -= .01
    mujoco.mj_forward(env.mj_model, data)
    contacts = np.asarray(data.sensordata)[
        np.asarray(env._foot_contact_sensor_adrs)
    ] > 0
    np.testing.assert_array_equal(contacts, np.ones((2, 4), dtype=bool))
    toe, heel = env._get_foot_clearances(mjx.put_data(env.mj_model, data))
    np.testing.assert_allclose(toe, -.01, atol=1e-6)
    np.testing.assert_allclose(heel, -.01, atol=1e-6)


if __name__ == '__main__':
  unittest.main()
