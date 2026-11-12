"""Recovery disturbance timing, batching, and physical/IMU consistency."""

import unittest
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from mujoco_playground._src.locomotion.rick import joystick


class YawDisturbanceTest(unittest.TestCase):
    def make_env(self, **overrides):
        env = object.__new__(joystick.Joystick)
        env._ctrl_dt = 0.02
        config = joystick.default_config()
        for name, value in overrides.items():
            config[name] = value
        env._configure_yaw_disturbance(config)
        return env

    def rollout(self, env, key, steps=500):
        def step(state, _):
            return env._step_yaw_disturbance(state)
        return jax.lax.scan(step, env._reset_yaw_disturbance(key), None, steps)[1]

    def test_exact_pulse_length_and_rest_without_lingering_torque(self):
        env = self.make_env(
            yaw_disturbance_probability=1.0,
            yaw_disturbance_start_range=(0.04, 0.04),
            yaw_disturbance_interval_range=(0.06, 0.06),
            yaw_disturbance_duration=0.04,
            yaw_disturbance_torque_range=(0.002, 0.002),
            yaw_disturbance_ramp_seconds=0.0,
        )
        torque = jax.jit(lambda key: self.rollout(env, key, 12))(
            jax.random.PRNGKey(0))
        np.testing.assert_allclose(
            torque, np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]) * .002)

    def test_disabled_episodes_and_batched_reproducibility(self):
        keys = jax.random.split(jax.random.PRNGKey(3), 64)
        env = self.make_env()
        run = jax.jit(jax.vmap(lambda key: self.rollout(env, key)))
        first = np.asarray(run(keys))
        np.testing.assert_array_equal(first, run(keys))
        disturbed = np.any(first != 0, axis=1)
        self.assertGreater(disturbed.sum(), 16)
        self.assertLess(disturbed.sum(), 48)
        self.assertTrue(np.any(first > 0) and np.any(first < 0))
        self.assertTrue(np.all(first[:, :100] == 0))
        self.assertLessEqual(np.abs(first).max(), .002)
        disabled = self.make_env(yaw_disturbance_probability=0.0)
        np.testing.assert_array_equal(self.rollout(disabled, keys[0]), 0)

    def test_ramp_is_sampled_at_pulse_start_and_reaches_full_strength(self):
        env = self.make_env(
            yaw_disturbance_probability=1.0,
            yaw_disturbance_start_range=(.02, .02),
            yaw_disturbance_interval_range=(.02, .02),
            yaw_disturbance_duration=.04,
            yaw_disturbance_torque_range=(.002, .002),
            yaw_disturbance_ramp_seconds=.10,
        )
        torque = np.asarray(self.rollout(env, jax.random.PRNGKey(0), 12))
        np.testing.assert_allclose(torque[1:3], .002 * .4)
        np.testing.assert_allclose(torque[7:9], .002)

    def test_rejects_invalid_config(self):
        for name, value in (
            ('yaw_disturbance_probability', 1.1),
            ('yaw_disturbance_duration', 0.0),
            ('yaw_disturbance_ramp_seconds', -1.0),
            ('yaw_disturbance_torque_range', (0.1, -0.1)),
            ('yaw_disturbance_interval_range', (0.0, 1.0)),
            ('yaw_disturbance_start_range', (float('nan'), 1.0)),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.make_env(**{name: value})

    def test_legacy_configs_opt_out_but_allow_explicit_overrides(self):
        for overrides, expected in ((None, 0.0), (
            {'yaw_disturbance_probability': 1.0,
             'yaw_disturbance_torque_range': [-.002, -.002]}, 1.0
        )):
            config = joystick.default_config()
            for name in joystick._YAW_DISTURBANCE_DEFAULTS:
                del config[name]
            config.lock()
            # Stop before model loading; test the real initializer's config
            # migration/override ordering without fetching assets.
            with mock.patch.object(joystick.consts, 'task_to_xml',
                                   side_effect=RuntimeError('model boundary')):
                with self.assertRaisesRegex(RuntimeError, 'model boundary'):
                    joystick.Joystick(config=config, config_overrides=overrides)
            self.assertEqual(config.yaw_disturbance_probability, expected)

    def test_physical_world_yaw_is_visible_to_imu_for_both_signs(self):
        # No contacts: isolate axis/sign and continuous gyro integration from
        # gait skill. The real model/contact response is checked separately.
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <option timestep=".002" gravity="0 0 0"/>
          <worldbody><body><freejoint/>
            <geom type="sphere" size=".03" mass=".25"/>
            <site name="imu"/>
          </body></worldbody></mujoco>''')
        env = self.make_env()
        env._imu_site_id = 0
        env._madgwick_beta = 0.1
        for sign in (-1, 1):
            data = mujoco.MjData(model)
            quat = jp.array([1., 0., 0., 0.])
            for step in range(20):
                data.xfrc_applied[:] = 0
                if step < 8:
                    data.xfrc_applied[1, 5] = sign * .0005
                for _ in range(10):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                # Native site_xmat is flat; the environment uses MJX matrices.
                signals = SimpleNamespace(
                    qpos=jp.array(data.qpos), qvel=jp.array(data.qvel),
                    site_xmat=jp.array(data.site_xmat.reshape(-1, 3, 3)),
                )
                gravity, gyro = env._get_imu_signals(signals)
                quat = env._update_attitude_estimate(quat, gyro, -gravity)
            measured = float(env._heading_from_quat(quat))
            actual = float(env._heading_from_quat(jp.array(data.qpos[3:7])))
            self.assertGreater(sign * measured, 0.1)
            self.assertAlmostEqual(measured, actual, delta=.01)
            np.testing.assert_allclose(data.qvel[:5], 0, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
