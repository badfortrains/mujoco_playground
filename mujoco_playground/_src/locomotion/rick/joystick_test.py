"""Tests for Rick's deployment-compatible controller models."""

from types import SimpleNamespace

from absl.testing import absltest
import jax
from jax import numpy as jp
import numpy as np
import mujoco

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
    self.env._heading_observation = False

  def test_default_retraining_config(self):
    config = joystick.default_config()
    self.assertEqual(config.target_velocity, 0.068)
    self.assertEqual(config.action_scale, 0.40)
    self.assertEqual(config.step_frequency, 1.2)
    self.assertEqual(config.command_history_length, 12)
    self.assertTrue(config.heading_observation)
    self.assertTrue(config.gyro_bias_calibration)
    self.assertEqual(config.gyro_bias_calibration_samples, 100)
    self.assertEqual(config.cycle_sideways_velocity_cost_weight, 0.5)
    self.assertEqual(config.cycle_yaw_rate_cost_weight, 4.0)
    self.assertEqual(tuple(config.action_delay_range), (0, 4))
    self.assertEqual(tuple(config.foot_friction_range), (0.20, 0.60))
    self.assertEqual(
        tuple(config.center_of_mass_offset_scale),
        (0.005, 0.005, 0.0020),
    )
    self.assertEqual(tuple(config.servo_stiffness_range), (1.0, 1.0))
    self.assertEqual(
        tuple(config.servo_frictionloss_range), (0.010, 0.025)
    )
    self.assertEqual(tuple(config.servo_backlash_range_us), (5.0, 20.0))
    self.assertEqual(config.action_noise_scale, 0.01)
    self.assertEqual(
        tuple(config.root_position_noise_scale),
        (0.002, 0.002, 0.0007),
    )
    self.assertEqual(config.joint_position_noise_scale, 0.03)
    self.assertEqual(config.swing_foot_forward_distance, 0.030)
    self.assertEqual(config.swing_foot_height, 0.0081)
    self.assertTrue(config.foot_height_uses_sole_clearance)
    self.assertEqual(config.foot_height_reward, 'supported_clearance')
    self.assertEqual(config.foot_phase_reward_weight, 4.0)
    self.assertEqual(config.stance_foot_height_cost_weight, 0.0)
    self.assertEqual(config.contact_phase_cost_weight, 1.0)
    self.assertEqual(config.swing_foot_forward_tracking_sigma, 1.0e-4)
    self.assertEqual(config.flight_cost_weight, 1.0)
    self.assertEqual(config.base_height_tracking_sigma, 1.8e-4)
    self.assertEqual(tuple(config.healthy_z_range), (0.034, 0.135))
    self.assertEqual(config.velocity_tracking_weight, 2.0)
    self.assertEqual(config.yaw_rate_cost_weight, 0.4)
    self.assertEqual(config.heading_cost_weight, 2.0)
    self.assertEqual(config.orientation_cost_weight, 0.3)

  def _supported_clearance_reward(self):
    self.env._foot_height_reward = 'supported_clearance'
    self.env._foot_phase_reward_weight = 4.0
    self.env._stance_foot_height_cost_weight = 0.5
    self.env._contact_phase_cost_weight = 0.0
    self.env._flight_cost_weight = 1.0
    self.env._swing_foot_height = 0.0081
    return jax.jit(self.env._get_foot_height_rewards)

  def test_supported_clearance_requires_opposite_contact_on_both_sides(self):
    reward = self._supported_clearance_reward()
    h = self.env._swing_foot_height
    for swing_index, phase in ((0, jp.pi / 2), (1, -jp.pi / 2)):
      for fraction, expected in ((0.0, 0.0), (0.5, 2.0), (1.0, 4.0), (2.0, 4.0)):
        heights = jp.zeros(2).at[swing_index].set(fraction * h)
        contacts = jp.array([True, True]).at[swing_index].set(False)
        values = reward(jp.array(phase), heights, contacts, heights, heights)
        np.testing.assert_allclose(values, [expected, 0, 0], atol=1e-6)
        # Contact on the swing foot cannot substitute for opposite-foot support.
        values = reward(jp.array(phase), heights, ~contacts, heights, heights)
        np.testing.assert_allclose(values, [0, 0, 0], atol=1e-6)
      heights = jp.full(2, h)
      values = reward(jp.array(phase), heights, jp.array([False, False]),
                      heights, heights)
      np.testing.assert_allclose(values, [-1.5, 0, 0], atol=1e-6)

  def test_supported_clearance_preserves_unloading_with_or_without_support(self):
    reward = self._supported_clearance_reward()
    h = self.env._swing_foot_height
    for contact in ((True, True), (False, True), (True, False), (False, False)):
      contacts = jp.array(contact)
      def score(left, right):
        heights = jp.array([left, right])
        return float(reward(jp.array(jp.pi / 2), heights, contacts,
                            heights, heights)[0])
      # The -4 to -2 improvement survives even if support is absent.
      self.assertAlmostEqual(score(-h / 2, -h) - score(-h, -h), 2.0, places=5)
      self.assertAlmostEqual(score(-h, -h / 2), score(-h, -h), places=5)
      self.assertAlmostEqual(score(0, 0), 0.0 if any(contact) else -1.0, places=5)

  def test_supported_clearance_rejects_tilt_and_penalizes_flight_at_exchange(self):
    reward = self._supported_clearance_reward()
    h = self.env._swing_foot_height
    toe, heel = jp.array([h, 0.0]), jp.zeros(2)
    values = reward(jp.array(jp.pi / 2), toe, jp.array([False, True]), toe, heel)
    np.testing.assert_allclose(values, [0, 0, 0], atol=1e-6)
    for phase in (0.0, jp.pi, 2 * jp.pi):
      heights = jp.full(2, h)
      values = reward(jp.array(phase), heights, jp.array([False, False]),
                      heights, heights)
      np.testing.assert_allclose(values, [-1, 0, 0], atol=1e-6)
    heights = jp.array([h, 0.0])
    values = reward(jp.array(jp.pi / 4), heights, jp.array([False, True]),
                    heights, heights)
    np.testing.assert_allclose(values, [2, 0, 0], atol=1e-6)
    np.testing.assert_allclose(
        self.env._desired_foot_heights(jp.array(jp.pi / 4)), [h, 0.0])
    np.testing.assert_allclose(
        self.env._desired_foot_heights(jp.array(-jp.pi / 4)), [0.0, h])

  def test_contact_phase_penalty_prefers_correct_support_on_both_sides(self):
    reward = self._supported_clearance_reward()
    self.env._contact_phase_cost_weight = 1.0
    self.env._stance_foot_height_cost_weight = 0.0
    heights = jp.zeros(2)
    for swing_index, phase in ((0, jp.pi / 2), (1, -jp.pi / 2)):
      correct = jp.array([True, True]).at[swing_index].set(False)
      for contacts, expected in (
          (correct, 0.0), (~correct, -2.0),
          (jp.array([True, True]), -1.0),
          (jp.array([False, False]), -2.0),  # Contact error + flight penalty.
      ):
        values = reward(jp.array(phase), heights, contacts, heights, heights)
        np.testing.assert_allclose(values, [expected, 0, 0], atol=1e-6)
      lifted = heights.at[swing_index].set(self.env._swing_foot_height)
      values = reward(jp.array(phase), lifted, correct, lifted, lifted)
      np.testing.assert_allclose(values, [4, 0, 0], atol=1e-6)
      both_lifted = jp.full(2, self.env._swing_foot_height)
      values = reward(jp.array(phase), both_lifted, jp.array([False, False]),
                      both_lifted, both_lifted)
      np.testing.assert_allclose(values, [-2, 0, 0], atol=1e-6)

  def test_contact_phase_penalty_fades_at_exchange_and_preserves_unloading(self):
    reward = self._supported_clearance_reward()
    self.env._contact_phase_cost_weight = 1.0
    self.env._stance_foot_height_cost_weight = 0.0
    heights = jp.zeros(2)
    for phase, expected in ((0.0, 0.0), (jp.pi / 4, -0.5),
                            (jp.pi, 0.0), (2 * jp.pi, 0.0)):
      values = reward(jp.array(phase), heights, jp.array([True, True]),
                      heights, heights)
      np.testing.assert_allclose(values, [expected, 0, 0], atol=1e-6)
    for contact in ((True, True), (False, True), (True, False), (False, False)):
      scores = []
      for fraction in (-1.0, -0.5):
        h = jp.array([fraction, -1.0]) * self.env._swing_foot_height
        scores.append(float(reward(jp.array(jp.pi / 2), h, jp.array(contact),
                                   h, h)[0]))
      self.assertAlmostEqual(scores[1] - scores[0], 2.0, places=5)
    values = reward(jp.array(0.0), heights, jp.array([False, False]),
                    heights, heights)
    np.testing.assert_allclose(values, [-1, 0, 0], atol=1e-6)
    # Raising only a toe still earns no clearance bonus.
    toe = jp.array([self.env._swing_foot_height, 0.0])
    values = reward(jp.array(jp.pi / 2), toe, jp.array([False, True]),
                    toe, heights)
    np.testing.assert_allclose(values, [0, 0, 0], atol=1e-6)

  def test_phase_clearance_rewards_only_scheduled_lift_and_caps_credit(self):
    self.env._foot_height_reward = 'phase_clearance'
    self.env._foot_phase_reward_weight = 4.0
    self.env._stance_foot_height_cost_weight = 0.5
    self.env._swing_foot_height = 0.0081
    reward = jax.jit(self.env._get_foot_height_rewards)
    h = self.env._swing_foot_height
    def score(left, right=0.0, phase=jp.pi / 2):
      heights = jp.array([left, right])
      values = reward(jp.array(phase), heights, jp.array([True, True]),
                      heights, heights)
      self.assertEqual(float(values[1]), 0.0)
      self.assertEqual(float(values[2]), 0.0)
      return float(values[0])
    self.assertAlmostEqual(score(0), 0.0, places=5)
    self.assertAlmostEqual(score(h / 2), 2.0, places=5)
    self.assertAlmostEqual(score(h), 4.0, places=5)
    self.assertAlmostEqual(score(2 * h), 4.0, places=5)
    self.assertAlmostEqual(score(0, h), -0.5, places=5)
    self.assertAlmostEqual(score(h, h), 3.5, places=5)
    self.assertAlmostEqual(score(h, 2 * h), 3.0, places=5)
    self.assertAlmostEqual(score(0, h, phase=3 * jp.pi / 2), 4.0, places=5)
    # Under the floor, only unloading the scheduled swing foot earns progress.
    self.assertGreater(score(-h / 2, -h), score(-h, -h))
    self.assertAlmostEqual(score(-h, -h / 2), score(-h, -h), places=5)
    # Smooth phase weighting fades to zero at foot exchanges.
    self.assertAlmostEqual(score(h, 0, phase=jp.pi / 4), 2.0, places=5)
    for phase in (0.0, jp.pi, 2 * jp.pi):
      self.assertAlmostEqual(score(h, h, phase=phase), 0.0, places=5)

  def test_phase_clearance_uses_lowest_corner_and_reports_clearance_cap(self):
    self.env._foot_height_reward = 'phase_clearance'
    self.env._foot_phase_reward_weight = 4.0
    self.env._stance_foot_height_cost_weight = 0.5
    self.env._swing_foot_height = 0.0081
    reward = jax.jit(self.env._get_foot_height_rewards)
    toe = jp.array([0.0081, 0.0])
    for contact in (True, False):
      values = reward(jp.array(jp.pi / 2), toe,
                      jp.array([contact, True]), toe, jp.zeros(2))
      self.assertAlmostEqual(float(values[0]), 0.0, places=5)
    np.testing.assert_allclose(
        self.env._desired_foot_heights(jp.array(jp.pi / 4)), [0.0081, 0.0])
    np.testing.assert_allclose(
        self.env._desired_foot_heights(jp.array(5 * jp.pi / 4)), [0.0, 0.0081])

  def test_linear_height_penalty_rewards_progress_and_penalizes_overshoot(self):
    self.env._foot_height_reward = 'linear'
    self.env._foot_phase_reward_weight = 4.0
    self.env._swing_foot_height = 0.0081
    reward = jax.jit(self.env._get_foot_height_rewards)
    h = self.env._swing_foot_height
    def score(left, right=0.0):
      heights = jp.array([left, right])
      values = reward(jp.array(jp.pi / 2), heights,
                      jp.array([True, True]), heights, heights)
      self.assertEqual(float(values[1]), 0.0)
      self.assertEqual(float(values[2]), 0.0)
      return float(values[0])
    self.assertAlmostEqual(score(0), -4.0, places=5)
    self.assertAlmostEqual(score(h / 2), -2.0, places=5)
    self.assertAlmostEqual(score(h), 0.0, places=5)
    self.assertAlmostEqual(score(2 * h), -4.0, places=5)
    self.assertLess(score(-h), score(0))
    self.assertLess(score(h, h), score(h, 0))
    # Far-off errors still have a nonzero improvement signal.
    self.assertGreater(score(-9 * h), score(-10 * h))

  def test_linear_height_penalty_tracks_phase_and_lowest_corner(self):
    self.env._foot_height_reward = 'linear'
    self.env._foot_phase_reward_weight = 4.0
    self.env._swing_foot_height = 0.0081
    reward = jax.jit(self.env._get_foot_height_rewards)
    for phase in (0.0, 0.4, 1.57, 3.14, 4.71, 6.28):
      desired = self.env._desired_foot_heights(jp.array(phase))
      for contact in (True, False):
        actual = reward(jp.array(phase), desired, jp.array([contact, contact]),
                        desired, desired)
        self.assertAlmostEqual(float(actual[0]), 0.0, places=5)
    phase = jp.array(jp.pi / 2)
    toe = jp.array([0.0081, 0.0])
    heel = jp.zeros(2)
    actual = reward(phase, toe, jp.array([False, True]), toe, heel)
    self.assertAlmostEqual(float(actual[0]), -4.0, places=5)
    # At the opposite half-cycle the other foot must rise.
    actual = reward(phase + jp.pi, toe, jp.array([False, True]), toe, toe)
    self.assertAlmostEqual(float(actual[0]), -8.0, places=5)

  def test_legacy_height_rewards_keep_all_three_terms(self):
    self.env._foot_height_reward = 'legacy'
    self.env._foot_phase_reward_weight = 1.5
    self.env._swing_foot_height = 0.012
    self.env._foot_height_tracking_sigma = 1e-4
    self.env._swing_lift_reward_weight = 2.0
    self.env._swing_lift_contact_depth = 0.005
    self.env._swing_lift_target_clearance = 0.010
    self.env._swing_airborne_reward_weight = 1.0
    self.env._swing_foot_clearance = 0.007
    phase = jp.array(jp.pi / 2)
    heights = jp.array([0.006, 0.0])
    values = jax.jit(self.env._get_foot_height_rewards)(
        phase, heights, jp.array([False, True]), heights, heights
    )
    np.testing.assert_allclose(values, [1.5 * np.exp(-0.36),
                                       2 * 0.011 / 0.015, 0.006 / 0.007],
                               rtol=1e-6)

  def test_lift_reward_increases_before_contact_release_and_is_bounded(self):
    self.env._swing_lift_reward_weight = 2.0
    self.env._swing_lift_contact_depth = 0.005
    self.env._swing_lift_target_clearance = 0.010
    reward = jax.jit(self.env._get_swing_lift_reward)
    phase = jp.array(jp.pi / 2)
    # Even below the floor, progressively unloading the sole earns credit.
    heights = [-0.010, -0.005, -0.004, -0.002, 0.0, 0.005, 0.010, 0.020]
    rewards = [float(reward(phase, jp.array([h, 0.0]), jp.array([h, 0.0])))
               for h in heights]
    self.assertEqual(rewards[0], 0.0)
    self.assertEqual(rewards[1], 0.0)
    self.assertTrue(np.all(np.diff(rewards[1:-1]) > 0))
    self.assertAlmostEqual(rewards[-2], 2.0)
    self.assertAlmostEqual(rewards[-1], 2.0)

  def test_lift_reward_requires_the_scheduled_foot_and_lowest_corner(self):
    self.env._swing_lift_reward_weight = 2.0
    self.env._swing_lift_contact_depth = 0.005
    self.env._swing_lift_target_clearance = 0.010
    reward = jax.jit(self.env._get_swing_lift_reward)
    phase = jp.array(jp.pi / 2)
    toe = jp.array([0.010, -0.005])
    heel = jp.array([-0.005, -0.005])
    self.assertAlmostEqual(float(reward(phase, toe, heel)), 0.0)
    for offset in (0.0, float(jp.pi)):
      clear = toe if offset == 0.0 else toe[::-1]
      self.assertAlmostEqual(float(reward(phase + offset, clear, clear)), 2.0)
      self.assertAlmostEqual(float(reward(phase + offset, clear[::-1], clear[::-1])), 0.0)
    self.assertLess(float(reward(jp.array(0.0), toe, toe)), 1e-7)

  def test_airborne_reward_requires_contact_release_and_both_ends_clear(self):
    self.env._swing_airborne_reward_weight = 1.0
    self.env._swing_foot_clearance = 0.005
    reward = jax.jit(self.env._get_swing_airborne_reward)
    phase = jp.array(jp.pi / 2)
    clear = jp.array([0.005, 0.0])
    airborne = jp.array([False, True])
    self.assertAlmostEqual(float(reward(phase, airborne, clear, clear)), 1.0)
    self.assertEqual(float(reward(phase, jp.array([True, True]), clear, clear)), 0.0)
    for toe, heel in [(0.0, 0.005), (0.005, 0.0), (0.005, -0.001)]:
      self.assertEqual(float(reward(
          phase, airborne, jp.array([toe, 0.0]), jp.array([heel, 0.0])
      )), 0.0)
    self.assertAlmostEqual(float(reward(phase, airborne, clear / 2, clear)), 0.5)
    self.assertAlmostEqual(float(reward(phase, airborne, clear * 2, clear * 2)), 1.0)
    # A lifted stance foot cannot earn the swing foot's reward.
    self.assertAlmostEqual(float(reward(phase, airborne[::-1], clear[::-1], clear[::-1])), 0.0)
    self.assertAlmostEqual(float(reward(phase + jp.pi, airborne[::-1], clear[::-1], clear[::-1])), 1.0)
    for boundary in (0.0, float(jp.pi), 2 * float(jp.pi)):
      for offset in (-1e-4, 0.0, 1e-4):
        self.assertLess(float(reward(
            jp.array(boundary + offset), jp.array([False, False]),
            jp.ones(2) * 0.005, jp.ones(2) * 0.005,
        )), 1e-7)

  def test_sole_corners_follow_loaded_mesh_and_box_transforms(self):
    vertices = (np.indices((2, 2, 2)).reshape(3, -1).T * 2 - 1) * [0.0375, 0.03, 0.003]
    vertex_text = ' '.join(str(x) for x in vertices.flat)
    self.env._mj_model = mujoco.MjModel.from_xml_string(f'''
      <mujoco><asset><mesh name="sole" vertex="{vertex_text}"/></asset>
      <worldbody>
        <body pos="0.1 0.2 0.3" euler="10 20 30">
          <geom name="left_foot_collision" type="mesh" mesh="sole" pos=".01 .02 .003"/>
          <site name="left_foot_center" pos=".01 .02 0"/>
        </body>
        <body pos="-.1 -.2 .4" euler="-15 25 -35">
          <geom name="right_foot_collision" type="box" size=".0375 .03 .003" pos=".01 .02 .003"/>
          <site name="right_foot_center" pos=".01 .02 0"/>
        </body>
      </worldbody></mujoco>''')
    self.env._feet_site_ids = jp.array([0, 1])
    data = mujoco.MjData(self.env._mj_model)
    mujoco.mj_forward(self.env._mj_model, data)
    corners = self.env._get_sole_corners(data)
    expected = [[-.0375, -.03, 0], [.0375, -.03, 0],
                [-.0375, .03, 0], [.0375, .03, 0]]
    np.testing.assert_allclose(corners, [expected, expected], atol=1e-8)

  def test_clearance_detects_pitch_and_roll_despite_raised_foot_center(self):
    self.env._feet_site_ids = jp.array([0, 1])
    self.env._sole_corners = jp.array([
        [[-.0375, -.0375, 0], [.0375, -.0375, 0],
         [-.0375, .0375, 0], [.0375, .0375, 0]],
    ] * 2)
    angle = np.arcsin(.010 / .0375)
    c, s = np.cos(angle), np.sin(angle)
    # Pitch left onto its toe; roll right onto one lateral edge.
    rotations = jp.array([
        [[1, 0, 0], [0, c, -s], [0, s, c]],
        [[c, 0, s], [0, 1, 0], [-s, 0, c]],
    ])
    def clearance(z):
      return self.env._get_foot_clearances(SimpleNamespace(
          site_xmat=rotations, site_xpos=jp.array([[0, 0, z], [0, 0, z]]),
      ))
    toe, heel = jax.jit(clearance)(.010)
    np.testing.assert_allclose(toe, [0, 0], atol=1e-8)
    np.testing.assert_allclose(heel, [.020, 0], atol=1e-8)
    toe, heel = jax.jit(clearance)(.022)
    np.testing.assert_allclose(toe, [.012, .012], atol=1e-8)
    np.testing.assert_allclose(heel, [.032, .012], atol=1e-8)

  def test_foot_slip_cost_requires_contact_regardless_of_gait_phase(self):
    self.env._foot_slip_cost_weight = 20.0
    foot_velocities = jp.array([
        [0.1, 0.2, 3.0],
        [0.3, 0.4, 5.0],
    ])
    cost = self.env._get_foot_slip_cost(
        foot_velocities,
        jp.array([True, False]),
        jp.array([True, False]),
    )
    self.assertAlmostEqual(float(cost), 1.0, places=6)

    no_contact_cost = self.env._get_foot_slip_cost(
        foot_velocities,
        jp.array([False, False]),
        jp.array([False, False]),
    )
    self.assertEqual(float(no_contact_cost), 0.0)

    both_contact_cost = jax.jit(self.env._get_foot_slip_cost)(
        foot_velocities,
        jp.array([True, True]),
        jp.array([True, True]),
    )
    np.testing.assert_allclose(both_contact_cost, 6.0, rtol=1e-6)

    vertical_only_cost = self.env._get_foot_slip_cost(
        jp.array([[0.0, 0.0, 3.0], [0.0, 0.0, 5.0]]),
        jp.array([True, True]),
        jp.array([True, True]),
    )
    self.assertEqual(float(vertical_only_cost), 0.0)

  def test_foot_slip_excludes_airborne_travel_at_contact_transitions(self):
    self.env._foot_slip_cost_weight = 20.0
    cost = jax.jit(self.env._get_foot_slip_cost)
    velocities = jp.array([[0.2, 0.0, 0.0], [0.1, 0.0, 0.0]])
    # Left foot lands after airborne travel; right foot remains planted.
    touchdown_cost = cost(
        velocities, jp.array([False, True]), jp.array([True, True])
    )
    np.testing.assert_allclose(touchdown_cost, 0.2, rtol=1e-6)
    # Left foot lifts off; its airborne travel should likewise be exempt.
    liftoff_cost = cost(
        velocities, jp.array([True, True]), jp.array([False, True])
    )
    np.testing.assert_allclose(liftoff_cost, 0.2, rtol=1e-6)
    # Continued contact in the next interval must incur slip on both feet,
    # even if the gait clock says the left foot should be swinging.
    dragging_cost = cost(
        velocities, jp.array([True, True]), jp.array([True, True])
    )
    np.testing.assert_allclose(dragging_cost, 1.0, rtol=1e-6)

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
        atol=1e-6,
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

  def test_heading_observation_appends_estimated_relative_heading(self):
    self.env._heading_observation = True
    for angle in (0.0, np.pi / 2, np.pi * 2 + 0.3):
      obs = jax.jit(self.env._get_observation)(
          jp.zeros((12, 8)), jp.array([0., 0., -1.]), jp.zeros(3),
          jp.array(0.0), jp.array(angle),
      )
      self.assertEqual(obs.shape, (107,))
      np.testing.assert_allclose(obs[105:], [np.sin(angle), np.cos(angle)], atol=3e-7)

  def test_heading_uses_forward_axis_and_survives_wrap_and_quaternion_sign(self):
    from brax import math
    tilt = self.env._roll_pitch_quat(jp.array([0.2, -0.3]))
    for angle in (-3.2, -0.4, 0.0, 1.2, 3.2):
      yaw = jp.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
      q = math.quat_mul(yaw, tilt)
      forward = math.rotate(jp.array([0., -1., 0.]), q)
      expected = np.arctan2(forward[0], -forward[1])
      for sign in (1, -1):
        actual = self.env._heading_from_quat(sign * q)
        np.testing.assert_allclose(actual, expected, atol=3e-7)

  def test_cycle_mean_cancels_sway_and_retains_drift_without_startup_bias(self):
    self.env._cycle_window = 4
    self.env._cycle_weights = jp.ones(4) / 4
    update = jax.jit(self.env._update_cycle_velocity)
    history, count = jp.zeros((4, 2)), jp.array(0)
    for i in range(12):
      sway = jp.array([0.1, 0.6]) * (1 if i % 2 else -1)
      history, count, mean, ready = update(history, count, sway + jp.array([.02, .05]))
      self.assertEqual(bool(ready), i >= 3)
      np.testing.assert_allclose(mean, [.02, .05] if i >= 3 else [0, 0], atol=2e-8)
    # At a non-integral period, only the oldest interval is fractional.
    self.env._cycle_weights = jp.array([.5, 1., 1., 1.]) / 3.5
    _, _, mean, _ = self.env._update_cycle_velocity(
        jp.array([[9., 9.], [2., 0.], [4., 0.], [6., 0.]]),
        jp.array(4), jp.array([8., 0.]),
    )
    np.testing.assert_allclose(mean, [(1 + 4 + 6 + 8) / 3.5, 0], rtol=1e-6)

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
        [0.0, body_position[1] - desired[1], 0.0],
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

    # Correct swing placement must not hide a misplaced stance foot.
    misplaced_stance_reward, _, _ = self.env._get_swing_foot_forward_reward(
        correctly_placed_feet.at[1, 1].add(0.02),
        body_position,
        body_forward_xy,
        phase,
    )
    self.assertLess(float(misplaced_stance_reward), 0.05)

  def test_foot_forward_stance_allows_a_planted_foot(self):
    self.env._nominal_foot_forward_positions = jp.array([0.002, -0.003])
    self.env._swing_foot_forward_distance = 0.030
    # The left foot is in stance from pi to 2*pi. A body advancing evenly
    # through one stride should leave its target fixed in world coordinates.
    phases = jp.linspace(jp.pi, 2.0 * jp.pi, 9)
    targets = jax.jit(jax.vmap(self.env._desired_foot_forward_positions))(
        phases
    )
    body_travel = jp.linspace(0.0, 0.030, 9)
    np.testing.assert_allclose(targets[:, 0] + body_travel, 0.017, atol=1e-7)

  def test_foot_forward_targets_alternate_and_wrap_continuously(self):
    self.env._nominal_foot_forward_positions = jp.array([0.0, 0.0])
    self.env._swing_foot_forward_distance = 0.030
    target = jax.jit(self.env._desired_foot_forward_positions)
    np.testing.assert_allclose(target(0.0), [-0.015, 0.015], atol=1e-7)
    np.testing.assert_allclose(target(jp.pi), [0.015, -0.015], atol=1e-7)
    for phase in (0.0, jp.pi / 4.0, jp.pi / 2.0, jp.pi):
      np.testing.assert_allclose(
          target(phase + jp.pi), target(phase)[::-1], atol=1e-7
      )
      np.testing.assert_allclose(
          target(phase - 2.0 * jp.pi), target(phase), atol=1e-7
      )
    for boundary in (0.0, jp.pi, 2.0 * jp.pi):
      np.testing.assert_allclose(
          target(boundary - 1e-5), target(boundary + 1e-5), atol=1e-6
      )

  def test_foot_forward_velocity_is_continuous_at_contact_transitions(self):
    self.env._nominal_foot_forward_positions = jp.array([0.0, 0.0])
    self.env._swing_foot_forward_distance = 0.030
    derivative = jax.jit(jax.jacfwd(self.env._desired_foot_forward_positions))
    # Both feet switch phases together. Target velocity must approach the
    # same stance retraction velocity on each side of liftoff/touchdown.
    phase_rate = 2.0 * jp.pi * 1.2
    for boundary in (0.0, jp.pi, 2.0 * jp.pi):
      for offset in (-1e-5, 1e-5):
        np.testing.assert_allclose(
            derivative(jp.array(boundary + offset)) * phase_rate,
            [-0.072, -0.072],
            atol=5e-6,
        )

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
