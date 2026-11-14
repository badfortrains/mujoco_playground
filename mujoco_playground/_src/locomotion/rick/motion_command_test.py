"""Speed/steering contract and reward direction tests without asset downloads."""
import unittest

import jax
import jax.numpy as jp
import numpy as np

from mujoco_playground._src.locomotion.rick import joystick


class MotionCommandTest(unittest.TestCase):
    def setUp(self):
        self.env = object.__new__(joystick.Joystick)
        self.env._ctrl_dt = .02
        config = joystick.default_config()
        config.update(joystick.command_stage_overrides('full'))
        self.env._configure_motion_commands(config)
        self.env._configure_reward_shaping(config)
        self.env._heading_observation = True
        self.env._gyro_obs_scale = .25
        self.env._step_frequency = 1.5
        self.env._swing_foot_forward_reward_weight = 1.

    def test_observation_contract(self):
        history = jp.arange(96).reshape(12, 8) / 100.
        obs = jax.jit(self.env._get_observation)(
            history, jp.array([0., 0., -1.]), jp.array([1., 2., 3.]),
            jp.array(.4), jp.array(-.7), jp.array([-.03, .4]))
        self.assertEqual(obs.shape, (108,))
        np.testing.assert_allclose(obs[:96], history.flatten())
        np.testing.assert_allclose(obs[96:102], [0, 0, -1, .25, .5, .75])
        np.testing.assert_allclose(obs[102:], [np.sin(.4), np.cos(.4), -.03, .4,
                                             np.sin(-.7), np.cos(-.7)], atol=1e-7)

    def test_sampling_covers_standing_straight_pivot_reverse_and_both_turns(self):
        commands, ticks = jax.jit(jax.vmap(self.env._sample_motion_command))(
            jax.random.split(jax.random.PRNGKey(7), 1000))
        commands = np.asarray(commands)
        self.assertTrue(np.all(commands >= [-.040001, -.600001]))
        self.assertTrue(np.all(commands <= [.080001, .600001]))
        self.assertTrue(np.all((ticks >= 150) & (ticks <= 300)))
        for mask in [np.all(commands == 0, axis=1),
                     (commands[:, 0] == 0) & (commands[:, 1] != 0),
                     (commands[:, 0] != 0) & (commands[:, 1] == 0),
                     commands[:, 0] < 0, commands[:, 1] < 0, commands[:, 1] > 0]:
            self.assertGreater(mask.sum(), 50)

    def test_slew_reversal_and_resampling_do_not_jump(self):
        step = jax.jit(self.env._step_motion_command)
        command = jp.zeros(2)
        target = jp.array([.08, .6])
        key = jax.random.PRNGKey(1)
        for i in range(100):
            previous = command
            command, retained, _ = step(command, target, jp.array(100), key)
            np.testing.assert_allclose(retained, target)
            self.assertTrue(np.all(np.abs(command - previous) <= [.0010001, .0200001]))
        np.testing.assert_allclose(command, target, atol=1e-7)
        command, _, _ = step(command, jp.array([-.04, -.6]), jp.array(1), key)
        np.testing.assert_allclose(command, [.079, .58], atol=1e-7)
        previous = command
        command, target, ticks = step(command, target, jp.array(0), key)
        self.assertGreaterEqual(int(ticks), 149)
        self.assertTrue(np.all(np.abs(command - previous) <= [.0010001, .0200001]))

    def test_velocity_tracking_is_invariant_to_world_heading(self):
        for angle in [0., .7, 2., -3.]:
            f = jp.array([np.sin(angle), -np.cos(angle)])
            right = jp.array([f[1], -f[0]])
            velocity = jp.concatenate([.06 * f + .01 * right, jp.array([0.])])
            np.testing.assert_allclose(self.env._body_planar_velocity(velocity, f),
                                       [.06, .01], atol=1e-8)

    def test_forward_reverse_and_pivot_swing_reward(self):
        feet = jp.array([[.025, 0., 0.], [-.025, 0., 0.]])
        phase = jp.array(np.pi / 2)
        def score(command, velocities, yaw=0.):
            return float(self.env._get_swing_foot_forward_reward(
                feet, jp.zeros(3), jp.array([0., -1.]), phase,
                velocities, jp.zeros(3), jp.array(command), jp.array(yaw),
                jp.array([False, True]), jp.full(2, .005), jp.full(2, .005))[0])
        forward = jp.array([[0., -.10, 0.], [0., .05, 0.]])
        self.assertGreater(score([.05, 0.], forward), .7)
        self.assertLess(score([-.05, 0.], forward), -.7)
        self.assertGreater(score([-.05, 0.], -forward), .7)
        pivot = jp.array([[0., .025, 0.], [0., .0125, 0.]])
        self.assertGreater(score([0., .5], pivot), .7)
        self.assertLess(score([0., -.5], pivot), -.7)
        # During a real pivot the stance foot stays planted; only the lifted
        # swing foot earns progress credit in the rotating body frame.
        pivot_planted = jp.array([[0., .0375, 0.], [0., 0., 0.]])
        self.assertGreater(score([0., .5], pivot_planted, yaw=.5), .7)
        self.assertEqual(score([0., 0.], forward), 0.)

    def test_standing_blend_handles_reverse_and_pivot(self):
        for command, activity in [([0, 0], 0), ([.005, 0], .5),
                                   ([-.01, 0], 1), ([0, -.1], 1)]:
            self.assertAlmostEqual(float(self.env._walking_activity(jp.array(command))), activity)

    def test_bad_config_and_old_runs_are_rejected(self):
        for name, value in [('motion_command_version', 0),
                            ('forward_velocity_range', (.01, .08)),
                            ('yaw_rate_range', (1, -1)),
                            ('linear_acceleration_limit', 0),
                            ('command_resampling_time_range', (0, 1))]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                config = joystick.default_config()
                config[name] = value
                self.env._configure_motion_commands(config)


if __name__ == '__main__':
    unittest.main()
