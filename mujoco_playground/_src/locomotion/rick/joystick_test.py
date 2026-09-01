"""Tests for Rick's deployment-compatible controller models."""

from types import SimpleNamespace

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
    self.assertEqual(config.command_history_length, 12)
    self.assertEqual(tuple(config.action_delay_range), (0, 2))
    self.assertEqual(tuple(config.foot_friction_range), (0.20, 0.60))
    self.assertEqual(
        tuple(config.center_of_mass_offset_scale),
        (0.005, 0.005, 0.003),
    )
    self.assertEqual(tuple(config.servo_stiffness_range), (0.25, 0.80))
    self.assertEqual(
        tuple(config.servo_frictionloss_range), (0.010, 0.025)
    )
    self.assertEqual(tuple(config.servo_backlash_range_us), (10.0, 30.0))
    self.assertEqual(config.action_noise_scale, 0.01)
    self.assertEqual(config.joint_position_noise_scale, 0.03)
    self.assertEqual(config.swing_foot_forward_distance, 0.030)

  def test_foot_slip_cost_requires_stance_and_contact(self):
    self.env._foot_slip_cost_weight = 20.0
    foot_velocities = jp.array([
        [0.1, 0.2, 3.0],
        [0.3, 0.4, 5.0],
    ])
    desired_heights = jp.array([0.0, 0.0])

    cost = self.env._get_foot_slip_cost(
        foot_velocities,
        desired_heights,
        jp.array([True, False]),
    )
    self.assertAlmostEqual(float(cost), 1.0, places=6)

    no_contact_cost = self.env._get_foot_slip_cost(
        foot_velocities,
        desired_heights,
        jp.array([False, False]),
    )
    self.assertEqual(float(no_contact_cost), 0.0)

    swing_cost = self.env._get_foot_slip_cost(
        foot_velocities,
        jp.array([0.01, 0.01]),
        jp.array([True, True]),
    )
    self.assertEqual(float(swing_cost), 0.0)

  def test_com_offset_preserves_world_body(self):
    self.env._mjx_model = SimpleNamespace(
        body_ipos=jp.array([
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
    )

    body_ipos = self.env._body_ipos_with_com_offset(
        jp.array([0.1, -0.2, 0.3])
    )

    np.testing.assert_allclose(body_ipos[0], jp.zeros(3))
    np.testing.assert_allclose(
        body_ipos[1:],
        jp.array([
            [1.1, 1.8, 3.3],
            [4.1, 4.8, 6.3],
        ]),
        atol=1e-6,
    )

  def test_servo_backlash_absorbs_small_changes_and_reversals(self):
    backlash = jp.array([0.1])
    output = jp.array([0.0])

    output = self.env._apply_servo_backlash(
        output, jp.array([0.05]), backlash
    )
    np.testing.assert_allclose(output, jp.array([0.0]))

    output = self.env._apply_servo_backlash(
        output, jp.array([0.25]), backlash
    )
    np.testing.assert_allclose(output, jp.array([0.15]))

    # Reversing by less than the full 2 * backlash gap does not move output.
    output = self.env._apply_servo_backlash(
        output, jp.array([0.10]), backlash
    )
    np.testing.assert_allclose(output, jp.array([0.15]))

    output = self.env._apply_servo_backlash(
        output, jp.array([-0.10]), backlash
    )
    np.testing.assert_allclose(output, jp.array([0.0]))

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
    command_history = (
        jp.arange(96, dtype=jp.float32).reshape((12, 8)) / 96
    )
    gravity = jp.array([0.1, -0.2, -0.97])
    gyro = jp.array([1.0, 2.0, 3.0])
    observation = self.env._get_observation(
        command_history,
        gravity,
        gyro,
        jp.array(0.0),
    )

    self.assertEqual(observation.shape, (105,))
    np.testing.assert_allclose(observation[:96], command_history.flatten())
    np.testing.assert_allclose(observation[96:99], gravity)
    np.testing.assert_allclose(observation[99:102], 0.25 * gyro)
    np.testing.assert_allclose(observation[102:104], jp.array([0.0, 1.0]))
    self.assertAlmostEqual(float(observation[104]), 0.04)

  def test_swing_foot_forward_reward_requires_body_relative_placement(self):
    self.env._nominal_foot_forward_positions = jp.array([0.0, 0.0])
    self.env._swing_foot_forward_distance = 0.030
    self.env._swing_foot_forward_reward_weight = 1.0
    self.env._swing_foot_forward_tracking_sigma = 1.0e-4
    phase = jp.array(3.0 * jp.pi / 4.0)
    desired = self.env._desired_foot_forward_positions(phase)
    body_position = jp.array([0.0, -0.2, 0.1])
    body_forward_xy = jp.array([0.0, -1.0])
    correctly_placed_feet = jp.array([
        [0.0, body_position[1] - desired[0], 0.01],
        [0.0, body_position[1], 0.0],
    ])

    correct_reward, actual, _ = self.env._get_swing_foot_forward_reward(
        correctly_placed_feet,
        body_position,
        body_forward_xy,
        phase,
    )
    falling_reward, _, _ = self.env._get_swing_foot_forward_reward(
        correctly_placed_feet,
        body_position + jp.array([0.0, -0.02, 0.0]),
        body_forward_xy,
        phase,
    )

    self.assertAlmostEqual(float(correct_reward), 1.0, places=6)
    self.assertAlmostEqual(float(actual[0]), float(desired[0]), places=6)
    self.assertLess(float(falling_reward), 0.05)

  def test_imu_velocity_uses_site_position(self):
    self.env._imu_site_id = 1
    before = SimpleNamespace(site_xpos=jp.array([
        [9.0, 9.0, 9.0],
        [0.1, 0.2, 0.3],
    ]))
    after = SimpleNamespace(site_xpos=jp.array([
        [8.0, 8.0, 8.0],
        [0.14, 0.18, 0.31],
    ]))

    velocity = self.env._get_imu_linear_velocity_world(before, after)

    np.testing.assert_allclose(
        velocity,
        jp.array([2.0, -1.0, 0.5]),
        atol=1e-6,
    )

  def test_imu_signals_are_expressed_in_site_frame(self):
    self.env._imu_site_id = 0
    imu_xmat = jp.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    data = SimpleNamespace(
        qpos=jp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        qvel=jp.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0]),
        site_xmat=imu_xmat[None, ...],
    )

    gravity, gyro = self.env._get_imu_signals(data)

    np.testing.assert_allclose(gravity, jp.array([0.0, 0.0, -1.0]))
    np.testing.assert_allclose(gyro, jp.array([2.0, -1.0, 3.0]))


if __name__ == '__main__':
  absltest.main()
