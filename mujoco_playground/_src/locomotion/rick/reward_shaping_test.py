"""Walking reward incentives, changing-command windows, and curriculum stages."""
import unittest

import jax
import jax.numpy as jp
import numpy as np

from mujoco_playground._src.locomotion.rick import joystick


class WalkingRewardTest(unittest.TestCase):
    def setUp(self):
        self.env = object.__new__(joystick.Joystick)
        self.env._ctrl_dt = .02
        self.config = joystick.default_config()
        self.env._configure_reward_shaping(self.config)
        self.env._velocity_tracking_weight = self.config.velocity_tracking_weight
        self.env._tracking_sigma = self.config.tracking_sigma
        self.env._swing_foot_forward_reward_weight = 1.
        self.env._cycle_window = 4
        self.env._cycle_weights = jp.ones(4) / 4

    def test_cycle_tracking_accepts_oscillation_but_rejects_drift(self):
        history = jp.zeros((4, 3))
        count = jp.array(0)
        for i in range(8):
            error = .04 if i % 2 == 0 else -.04
            history, count, mean, ready = self.env._update_cycle_velocity(
                history, count, jp.array([0., 0., error]))
        total, instant, cycle = jax.jit(self.env._get_velocity_tracking_reward)(
            jp.array(error), mean[2], ready)
        self.assertAlmostEqual(float(cycle), 1.5, places=6)
        self.assertAlmostEqual(float(instant), .5 * np.exp(-.04**2/.0025), places=6)
        biased = self.env._get_velocity_tracking_reward(jp.array(.04), jp.array(.04), True)[0]
        self.assertGreater(float(total), 1.7)
        self.assertLess(float(biased), .3)

    def test_speed_jumps_make_ignoring_commands_less_rewarding(self):
        # A policy stuck at 45 mm/s misses both 30 and 60 by 15 mm/s.
        old_perfect = float(self.env._get_velocity_tracking_reward(0., 0., True)[0])
        old_midpoint = float(self.env._get_velocity_tracking_reward(.015, .015, True)[0])
        self.config.update(joystick.command_stage_overrides('speed_jumps'))
        self.env._configure_reward_shaping(self.config)
        self.env._tracking_sigma = self.config.tracking_sigma
        self.env._velocity_tracking_weight = self.config.velocity_tracking_weight
        perfect = float(self.env._get_velocity_tracking_reward(0., 0., True)[0])
        midpoint = float(self.env._get_velocity_tracking_reward(.015, .015, True)[0])
        self.assertAlmostEqual(perfect, 4.)
        self.assertGreater(old_midpoint / old_perfect, .6)
        self.assertLess(midpoint / perfect, .4)
        self.assertGreater(perfect - midpoint, 3 * (old_perfect - old_midpoint))
        # Periodic surges are still tolerated when the full-cycle error is zero.
        oscillating = float(self.env._get_velocity_tracking_reward(.04, 0., True)[0])
        self.assertGreater(oscillating, 3.5)

    def test_no_free_cycle_reward_during_history_warmup(self):
        reward, instant, cycle = self.env._get_velocity_tracking_reward(.05, 0., False)
        self.assertAlmostEqual(float(reward), 2 * np.exp(-1), places=6)
        self.assertEqual(float(cycle), 0)
        self.assertEqual(float(reward), float(instant))

    def test_changing_command_uses_matching_historical_target(self):
        # Exact tracking of a speed transition must have zero cycle error.
        history = jp.zeros((4, 3)); count = jp.array(0)
        for speed in [.01, .02, .03, .04]:
            history, count, mean, ready = self.env._update_cycle_velocity(
                history, count, jp.array([0., 0., speed - speed]))
        self.assertTrue(bool(ready))
        self.assertEqual(float(mean[2]), 0)
        self.assertEqual(float(self.env._get_velocity_tracking_reward(0., mean[2], ready)[0]), 2.)
        # Fractional oldest sample still spans exactly one physical period.
        self.env._cycle_weights = jp.array([.5, 1., 1., 1.]) / 3.5
        _, _, mean, _ = self.env._update_cycle_velocity(
            jp.array([[99., 0., 99.], [2., 0., 2.], [4., 0., 4.], [6., 0., 6.]]),
            jp.array(4), jp.array([8., 0., 8.]))
        np.testing.assert_allclose(mean, [19/3.5, 0, 19/3.5], rtol=1e-6)

    def progress(self, contacts=(False, True), toe=.005, heel=.005,
                 phase=np.pi/2, swing_velocity=-.1, stance_velocity=0., offset=0.):
        return self.env._get_swing_foot_forward_reward(
            jp.array([[.025, offset, .005], [-.025, 0., 0.]]), jp.zeros(3),
            jp.array([0., -1.]), jp.array(phase),
            jp.array([[0., swing_velocity, 0.], [0., stance_velocity, 0.]]),
            jp.zeros(3), jp.array([.05, 0.]), jp.array(0.),
            jp.array(contacts), jp.array([toe, 0.]), jp.array([heel, 0.]))[0]

    def test_grounded_rocking_dragging_and_flight_earn_no_progress(self):
        for contacts, toe, heel in [((True, True), .005, .005),
                                    ((False, True), .02, 0.),
                                    ((False, True), 0., .02),
                                    ((False, True), -.001, .01),
                                    ((False, False), .02, .02)]:
            self.assertEqual(float(self.progress(contacts, toe, heel)), 0.)
        # Placement alone and stance retraction no longer pay.
        self.assertAlmostEqual(float(self.progress(phase=0., offset=.04)), 0., places=6)
        self.assertEqual(float(self.progress(swing_velocity=0., stance_velocity=.2)), 0.)

    def test_clearance_credit_is_bounded_and_directional(self):
        low = self.progress(toe=.0025, heel=.0025)
        full = self.progress()
        self.assertAlmostEqual(float(low), .5 * float(full), places=6)
        self.assertGreater(float(full), .7)
        self.assertEqual(float(self.progress(toe=.02, heel=.02)), float(full))
        self.assertLess(float(self.progress(swing_velocity=.1)), -.7)

    def test_dense_lift_reward_exists_before_contact_breaks(self):
        config = self.config
        for attr in ('foot_phase_reward_weight', 'stance_foot_height_cost_weight',
                     'contact_phase_cost_weight', 'swing_sole_tilt_cost_weight',
                     'flight_cost_weight', 'swing_lift_reward_weight', 'swing_airborne_reward_weight'):
            setattr(self.env, '_'+attr, config[attr])
        self.env._natural_clearance_scale = .012
        self.env._natural_contact_depth = .005
        def lift(height):
            return self.env._get_foot_height_rewards(
                jp.array(np.pi/2), jp.array([True, True]),
                jp.array([height, 0.]), jp.array([height, 0.]))[1]
        self.assertGreater(float(lift(-.0005)), float(lift(-.002)))


class CommandCurriculumTest(unittest.TestCase):
    def make_env(self, stage, **overrides):
        env = object.__new__(joystick.Joystick)
        env._ctrl_dt = .02
        config = joystick.default_config()
        config.update(joystick.command_stage_overrides(stage))
        config.update(overrides)
        env._configure_motion_commands(config)
        return env, config

    def test_walk_is_default_and_holds_50_mm_s_after_timer_expires(self):
        config = joystick.default_config()
        self.assertEqual(config.command_stage, 'walk')
        env, _ = self.make_env('walk')
        commands, _ = jax.jit(jax.vmap(env._sample_motion_command))(
            jax.random.split(jax.random.PRNGKey(0), 128))
        np.testing.assert_allclose(commands[:, 0], .05)
        np.testing.assert_array_equal(commands[:, 1], 0.)
        _, target, ticks = env._step_motion_command(
            jp.array([.04, 0.]), jp.array([.04, 0.]), jp.array(0), jax.random.PRNGKey(1))
        np.testing.assert_allclose(target, [.04, 0.])
        self.assertEqual(int(ticks), 0)

    def test_stages_constrain_commands_and_exported_limits(self):
        for stage in ('speed', 'transitions', 'turns', 'full'):
            env, config = self.make_env(stage)
            command, _ = jax.jit(jax.vmap(env._sample_motion_command))(
                jax.random.split(jax.random.PRNGKey(2), 2000))
            command = np.asarray(command)
            self.assertTrue(np.all(command >= [config.forward_velocity_range[0]-1e-7, config.yaw_rate_range[0]-1e-7]))
            self.assertTrue(np.all(command <= [config.forward_velocity_range[1]+1e-7, config.yaw_rate_range[1]+1e-7]))
            if stage != 'full':
                self.assertTrue(np.all(command[:, 0] >= 0))
                self.assertFalse(np.any((command[:, 0] == 0) & (command[:, 1] != 0)))
            if stage in ('speed', 'transitions'):
                np.testing.assert_array_equal(command[:, 1], 0)
            if stage == 'speed':
                self.assertTrue(np.all(command[:, 0] >= .03))
            if stage == 'transitions':
                self.assertGreater(np.all(command == 0, axis=1).sum(), 150)

    def test_speed_jumps_alternate_extremes_with_smooth_ramps(self):
        env, config = self.make_env('speed_jumps')
        initial, ticks = jax.jit(jax.vmap(env._sample_motion_command))(
            jax.random.split(jax.random.PRNGKey(3), 256))
        initial = np.asarray(initial)
        low = np.isclose(initial[:, 0], .03)
        high = np.isclose(initial[:, 0], .06)
        self.assertTrue(np.all(low | high))
        self.assertGreater(low.sum(), 80)
        self.assertGreater(high.sum(), 80)
        np.testing.assert_array_equal(initial[:, 1], 0.)
        self.assertTrue(np.all((np.asarray(ticks) >= 200) & (np.asarray(ticks) <= 300)))
        target, timer = env._sample_motion_command(jax.random.PRNGKey(7))
        def one(carry, key):
            command, target, timer = env._step_motion_command(*carry, key)
            return (command, target, timer), (command, target)
        _, (commands, targets) = jax.jit(lambda: jax.lax.scan(
            one, (jp.zeros(2), target, timer),
            jax.random.split(jax.random.PRNGKey(11), 1000)))()
        commands, targets = np.asarray(commands), np.asarray(targets)
        changes = np.flatnonzero(abs(np.diff(targets[:, 0])) > .001) + 1
        self.assertGreaterEqual(len(changes), 3)
        levels = targets[np.r_[0, changes], 0]
        np.testing.assert_allclose(abs(np.diff(levels)), .03, atol=1e-7)
        holds = np.diff(np.r_[0, changes])
        self.assertTrue(np.all((holds >= 200) & (holds <= 301)))
        self.assertLessEqual(abs(np.diff(np.vstack([np.zeros(2), commands]), axis=0)).max(), .001001)
        np.testing.assert_array_equal(commands[:, 1], 0.)
        self.assertEqual(config.tracking_sigma, .0001)
        self.assertEqual(config.velocity_tracking_weight, 4.)

    def test_turns_keep_walking_and_strict_speed_tracking(self):
        env, config = self.make_env('turns')
        commands, ticks = jax.jit(jax.vmap(env._sample_motion_command))(
            jax.random.split(jax.random.PRNGKey(13), 2000))
        commands = np.asarray(commands)
        self.assertTrue(np.all(commands[:, 0] >= .03))
        self.assertTrue(np.all(commands[:, 0] <= .06))
        self.assertGreater((commands[:, 1] == 0).sum(), 800)
        self.assertGreater((commands[:, 1] < 0).sum(), 350)
        self.assertGreater((commands[:, 1] > 0).sum(), 350)
        self.assertLessEqual(abs(commands[:, 1]).max(), .15)
        self.assertTrue(np.all((np.asarray(ticks) >= 200) & (np.asarray(ticks) <= 300)))
        command, target, _ = env._step_motion_command(
            jp.array([.045, 0.]), jp.array([.045, .15]), jp.array(10),
            jax.random.PRNGKey(1))
        np.testing.assert_allclose(command, [.045, .02], atol=1e-7)
        np.testing.assert_allclose(target, [.045, .15])
        self.assertEqual(config.tracking_sigma, .0001)
        self.assertEqual(config.velocity_tracking_weight, 4.)
        self.assertEqual(config.cycle_velocity_tracking_fraction, .75)
        self.assertEqual(config.instant_velocity_tracking_sigma, .0025)

    def test_speed_levels_cover_all_three_targets_in_each_episode(self):
        env, config = self.make_env('speed_levels')
        commands, ticks = jax.jit(jax.vmap(env._sample_motion_command))(
            jax.random.split(jax.random.PRNGKey(5), 600))
        speeds = np.asarray(commands[:, 0])
        for speed in (.03, .045, .06):
            self.assertGreater(np.isclose(speeds, speed).sum(), 140)
        self.assertTrue(np.all(np.isclose(speeds[:, None], [.03, .045, .06]).any(axis=1)))
        np.testing.assert_array_equal(commands[:, 1], 0.)
        self.assertTrue(np.all((np.asarray(ticks) >= 200) & (np.asarray(ticks) <= 300)))
        target, timer = env._sample_motion_command(jax.random.PRNGKey(9))
        def one(carry, key):
            state = env._step_motion_command(*carry, key)
            return state, state[:2]
        _, (commands, targets) = jax.jit(lambda: jax.lax.scan(
            one, (jp.zeros(2), target, timer),
            jax.random.split(jax.random.PRNGKey(17), 1000)))()
        commands, targets = np.asarray(commands), np.asarray(targets)
        changes = np.flatnonzero(abs(np.diff(targets[:, 0])) > .001) + 1
        self.assertGreaterEqual(len(changes), 3)
        levels = np.array([.03, .045, .06])
        sequence = targets[np.r_[0, changes], 0]
        indices = abs(sequence[:, None] - levels).argmin(axis=1)
        np.testing.assert_array_equal(indices[1:], (indices[:-1] + 1) % 3)
        self.assertEqual(set(indices[:3]), {0, 1, 2})
        self.assertLessEqual(abs(np.diff(np.vstack([np.zeros(2), commands]), axis=0)).max(), .001001)
        np.testing.assert_array_equal(commands[:, 1], 0.)
        self.assertEqual(config.tracking_sigma, .0001)
        self.assertEqual(config.velocity_tracking_weight, 4.)

    def test_invalid_discrete_speed_profiles_rejected(self):
        for overrides in [dict(command_speed_levels=(.03,)),
                          dict(command_speed_levels=(.03, .03, .06)),
                          dict(command_speed_levels=(.06, .045, .03)),
                          dict(command_speed_levels=(.03, float('nan'), .06)),
                          dict(command_speed_levels=(.03, .09)),
                          dict(command_speed_levels=((.03, .06),)),
                          dict(command_resample_in_episode=False),
                          dict(command_alternate_speed_endpoints=True),
                          dict(command_mode_probabilities=(.15, 0, .85, 0)),
                          dict(yaw_rate_range=(-.1, .1))]:
            with self.subTest(overrides=overrides), self.assertRaises((ValueError, TypeError)):
                self.make_env('speed_levels', **overrides)
        # Fixed-speed validation turns off the schedule and holds each requested target.
        env, _ = self.make_env('speed_levels', command_resample_in_episode=False,
                               command_speed_levels=())
        _, target, _ = env._step_motion_command(jp.zeros(2), jp.array([.045, 0.]),
                                               jp.array(0), jax.random.PRNGKey(0))
        np.testing.assert_allclose(target, [.045, 0.])

    def test_alternation_rejects_standing_turns_and_disabled_resampling(self):
        for overrides in [dict(command_mode_probabilities=(.15, 0, .85, 0)),
                          dict(yaw_rate_range=(-.1, .1)),
                          dict(command_resample_in_episode=False),
                          dict(command_forward_velocity_range=(.03, .03)),
                          dict(command_alternate_speed_endpoints=1)]:
            with self.subTest(overrides=overrides), self.assertRaises((ValueError, TypeError)):
                self.make_env('speed_jumps', **overrides)

    def test_bad_probabilities_and_sample_bounds_fail(self):
        for overrides in [dict(command_mode_probabilities=(0, 0, 1, 1)),
                          dict(command_mode_probabilities=(0, -.1, 1.1, 0)),
                          dict(command_forward_velocity_range=(.05, .09)),
                          dict(command_resample_in_episode=1)]:
            with self.subTest(overrides=overrides), self.assertRaises((ValueError, TypeError)):
                self.make_env('walk', **overrides)


if __name__ == '__main__':
    unittest.main()
