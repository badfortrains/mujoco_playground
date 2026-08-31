"""Tests for Rick's deployment-compatible controller models."""

from absl.testing import absltest
import jax
from jax import numpy as jp
import numpy as np

from mujoco_playground._src.locomotion.rick import joystick


class JoystickTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # The filter and observation builders do not depend on a MuJoCo model.
    self.env = object.__new__(joystick.Joystick)
    self.env._ctrl_dt = 0.02
    self.env._madgwick_beta = 0.1
    self.env._gyro_obs_scale = 0.25
    self.env._target_velocity = 0.04

  def test_default_retraining_config(self):
    config = joystick.default_config()
    self.assertEqual(config.target_velocity, 0.04)
    self.assertEqual(config.action_scale, 0.40)
    self.assertEqual(config.step_frequency, 0.68)
    self.assertEqual(tuple(config.action_delay_range), (0, 2))
    self.assertEqual(config.joint_position_noise_scale, 0.03)

  def test_stationary_madgwick_update_stays_upright(self):
    identity = jp.array([1.0, 0.0, 0.0, 0.0])
    updated = jax.jit(self.env._update_attitude_estimate)(
        identity,
        jp.zeros(3),
        jp.array([0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(updated, identity, atol=1e-7)
    np.testing.assert_allclose(
        self.env._gravity_from_quat(updated),
        jp.array([0.0, 0.0, -1.0]),
        atol=1e-7,
    )

  def test_madgwick_estimate_converges_to_tilted_gravity(self):
    true_quat = self.env._roll_pitch_quat(
        jp.array([0.0, jp.deg2rad(10.0)])
    )
    true_gravity = self.env._gravity_from_quat(true_quat)
    accelerometer = -true_gravity
    estimated_quat = jp.array([1.0, 0.0, 0.0, 0.0])

    for _ in range(50):
      estimated_quat = self.env._update_attitude_estimate(
          estimated_quat,
          jp.zeros(3),
          accelerometer,
      )

    initial_error = jp.linalg.norm(
        true_gravity - jp.array([0.0, 0.0, -1.0])
    )
    final_error = jp.linalg.norm(
        true_gravity - self.env._gravity_from_quat(estimated_quat)
    )
    self.assertLess(float(final_error), 0.01 * float(initial_error))

  def test_prewarmed_attitude_starts_at_body_tilt(self):
    root_quat = self.env._roll_pitch_quat(
        jp.array([jp.deg2rad(7.0), jp.deg2rad(-11.0)])
    )
    imu_quat = jax.jit(self.env._prewarmed_attitude_estimate)(root_quat)

    np.testing.assert_allclose(
        self.env._gravity_from_quat(imu_quat),
        self.env._gravity_from_quat(root_quat),
        atol=1e-7,
    )
    self.assertGreater(float(jp.linalg.norm(imu_quat[1:])), 0.01)

  def test_observation_keeps_firmware_layout(self):
    command_history = jp.arange(32, dtype=jp.float32).reshape((4, 8)) / 32
    gravity = jp.array([0.1, -0.2, -0.97])
    gyro = jp.array([1.0, 2.0, 3.0])
    observation = self.env._get_observation(
        command_history,
        gravity,
        gyro,
        jp.array(0.0),
    )

    self.assertEqual(observation.shape, (41,))
    np.testing.assert_allclose(observation[:32], command_history.flatten())
    np.testing.assert_allclose(observation[32:35], gravity)
    np.testing.assert_allclose(observation[35:38], 0.25 * gyro)
    np.testing.assert_allclose(observation[38:40], jp.array([0.0, 1.0]))
    self.assertAlmostEqual(float(observation[40]), 0.04)


if __name__ == '__main__':
  absltest.main()
