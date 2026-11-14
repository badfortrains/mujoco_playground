"""Rick walking task with observations suitable for the real robot.

The policy observation contains only values available at deployment:

* recent normalized servo commands (known by the controller),
* a body-frame gravity estimate and angular velocity from the LSM6DSO,
* the controller's gait clock, and
* the commanded body-forward speed and world-Z turn rate, and
* estimated heading error relative to the integrated turn command.

Joint state, root velocity, true heading, foot state, and contact state are used only
for rewards, termination, and diagnostics during training.
"""

from pathlib import Path

import jax
from jax import numpy as jp
from ml_collections import config_dict

import mujoco
from mujoco import mjx
import numpy as np
from brax import math

import mujoco_playground
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.rick import rick_constants as consts


_SERVO_US_PER_RADIAN = 2000.0 / 3.141592653589793

# Signed world-Z torque, sampled once per pulse. Old saved configs opt out.
_YAW_DISTURBANCE_DEFAULTS = {
    'yaw_disturbance_probability': 0.5,
    'yaw_disturbance_torque_range': (-0.002, 0.002),  # N m.
    'yaw_disturbance_duration': 0.16,  # s.
    'yaw_disturbance_start_range': (2.0, 4.0),  # s before first pulse.
    'yaw_disturbance_interval_range': (3.0, 6.0),  # s between pulses.
    # Within each episode, ramp from 25% to full strength over this time.
    # Set zero for full-strength evaluation; this is not a PPO-step curriculum.
    'yaw_disturbance_ramp_seconds': 10.0,
}


# Explicit stages advance between runs, after reviewing fixed-command rollouts.
# Exported ranges include zero because START ramps from rest. The sampling
# range can exclude near-zero speeds while recovering an actual walking gait.
def command_stage_overrides(stage: str) -> dict:
    stages = {
        'walk': ((0.05, 0.05), (0.0, 0.0), (0.0, 0.0, 1.0, 0.0), False),
        'speed': ((0.03, 0.06), (0.0, 0.0), (0.0, 0.0, 1.0, 0.0), False),
        'transitions': ((0.03, 0.06), (0.0, 0.0), (0.15, 0.0, 0.85, 0.0), True),
        'speed_jumps': ((0.03, 0.06), (0.0, 0.0), (0.0, 0.0, 1.0, 0.0), True),
        'speed_levels': ((0.03, 0.06), (0.0, 0.0), (0.0, 0.0, 1.0, 0.0), True),
        'turns': ((0.03, 0.06), (-0.15, 0.15), (0.0, 0.0, 0.50, 0.50), True),
        'full': ((-0.04, 0.08), (-0.6, 0.6), (0.15, 0.15, 0.30, 0.40), True),
    }
    if stage not in stages:
        raise ValueError(f'Unknown Rick command stage: {stage}')
    forward, yaw, probabilities, resample = stages[stage]
    overrides = {
        'command_stage': stage,
        'command_forward_velocity_range': forward,
        'forward_velocity_range': (min(0.0, forward[0]), forward[1]),
        'yaw_rate_range': yaw,
        # Order: standing, pivot, straight, combined motion.
        'command_mode_probabilities': probabilities,
        'command_resample_in_episode': resample,
        'command_alternate_speed_endpoints': stage == 'speed_jumps',
        'command_speed_levels': (0.03, 0.045, 0.06) if stage == 'speed_levels' else (),
    }
    if stage in ('speed_jumps', 'speed_levels', 'turns'):
        overrides.update({
            'command_resampling_time_range': (4.0, 6.0),
            'tracking_sigma': 0.0001,  # exp(-1) at 10 mm/s cycle error.
            'velocity_tracking_weight': 4.0,
        })
    return overrides


_REWARD_SHAPING_DEFAULTS = {
    'cycle_velocity_tracking_fraction': 0.75,
    'instant_velocity_tracking_sigma': 0.0025,
    'swing_progress_clearance': 0.005,  # Full credit at 5 mm; no gain above it.
}


def default_config() -> config_dict.ConfigDict:
    return config_dict.ConfigDict({
        # Simulation / control.
        'ctrl_dt': 0.02,
        'sim_dt': 0.002,
        'episode_length': 1000,
        'action_repeat': 1,
        'impl': 'jax',
        # Warp contact capacity is shared across the whole training batch;
        # constraint capacity is per world. Allow up to 16 contacts per robot.
        'naconmax': 16 * 8192,
        'njmax': 96,

        # Policy command and gait parameters.
        'target_velocity': 0.050,  # Default deployment speed, m/s.
        'motion_command_version': 1,
        **command_stage_overrides('walk'),
        'command_resampling_time_range': (3.0, 6.0),
        'linear_acceleration_limit': 0.05,  # m/s^2.
        'yaw_acceleration_limit': 1.0,  # rad/s^2.
        'standing_reward_weight': 4.0,
        'action_scale': 0.50,
        'step_frequency': 1.5,
        'auto_step_frequency': True,

        # Twelve commands = 240 ms of controller-known history at 50 Hz.  This
        # is the policy's only proxy for joint state on feedback-free servos.
        'command_history_length': 12,

        # Hidden actuator uncertainty.  These quantities are sampled once per
        # episode and independently for each servo unless noted otherwise.
        # The policy sees the command it sent, not the delayed or perturbed
        # command used by the simulation.
        'servo_response_range': (0.20, 0.45),
        'servo_center_offset_scale_us': 20.0,
        'servo_gain_range': (0.90, 1.10),
        # A shared scale captures battery-voltage variation.  It affects all
        # servo speed and torque limits together, while the other ranges retain
        # servo-to-servo variation.
        'servo_voltage_scale_range': (0.85, 1.00),
        'servo_strength_range': (0.85, 1.10),
        'servo_speed_range': (4.0, 8.0),  # rad/s before voltage scaling.
        # Stiffness scales the XML position-servo kp.  Lower stiffness plus dry
        # joint friction produces load-dependent droop and a breakaway error.
        'servo_stiffness_range': (1, 1),
        'servo_frictionloss_range': (0.010, 0.025),  # N m.
        # Command deadband is electronic; backlash is a stateful half-width at
        # the output.  A reversal traverses twice the sampled backlash value.
        'servo_deadband_range_us': (4.0, 12.0),
        'servo_backlash_range_us': (5.0, 20.0),
        'action_delay_range': (0, 4),  # Controller frames, inclusive.
        'action_noise_scale': 0.01,

        # Effective sliding friction between the feet and floor.  Sample one
        # shared value per episode so the policy cannot rely on the nominal
        # XML contact coefficient or on a single floor material.
        'foot_friction_range': (0.20, 0.60),

        # Shared local-frame offset applied to every non-world inertial frame.
        # Because Rick's link frames are aligned in the nominal pose, this
        # shifts the whole-robot COM by approximately the sampled offset while
        # preserving the relative mass contribution of each link.
        'center_of_mass_offset_scale': (0.005, 0.005, 0.0020),  # m.

        # Reset randomization.  Keep units separate: a single scale applied to
        # the free joint, quaternion, and leg joints is not physically useful.
        'root_position_noise_scale': (0.002, 0.002, 0.0007),  # m.
        'root_tilt_noise_scale': 0.08,  # rad
        'joint_position_noise_scale': 0.03,  # rad.
        'root_linear_velocity_noise_scale': 0.02,  # m/s.
        'root_angular_velocity_noise_scale': 0.10,  # rad/s.
        'joint_velocity_noise_scale': 0.10,  # rad/s.

        # Recovery practice after START, with half the episodes undisturbed.
        **_YAW_DISTURBANCE_DEFAULTS,

        # Deployment-like IMU model.  The simulated accelerometer includes
        # body acceleration and drives the same 6-DoF Madgwick update used by
        # the Pico.  Accelerometer values are in g and gyro values in rad/s.
        'accelerometer_noise_scale': 0.02,
        'accelerometer_bias_scale': 0.01,
        'accelerometer_range': 4.0,
        'gyro_noise_scale': 0.03,
        'gyro_bias_scale': 0.02,
        'gyro_obs_scale': 0.25,
        'madgwick_beta': 0.10,
        'heading_observation': True,
        # Finite-sample error of a stationary pre-START gyro average.
        'gyro_bias_calibration': True,
        'gyro_bias_calibration_samples': 100,

        # Task rewards.
        'velocity_tracking_weight': 2.0,
        'tracking_sigma': 0.0004,  # Applied to cycle-average speed error.
        **_REWARD_SHAPING_DEFAULTS,
        'sideways_tracking_sigma': 0.0025,  # Preserve the working sway tolerance.
        'sideways_velocity_cost_weight': 0.02,
        'yaw_rate_cost_weight': 0.4,
        # Average a full gait period BEFORE squaring: alternating sway cancels.
        'cycle_sideways_velocity_cost_weight': 2.0,
        'cycle_yaw_rate_cost_weight': 4.0,
        'heading_cost_weight': 3.0,
        'orientation_cost_weight': 1.2,
        'roll_pitch_rate_cost_weight': 0.02,
        'action_rate_cost_weight': 0.03,
        'action_acceleration_cost_weight': 0.01,
        # Penalize changes inside the episode's sampled MG90S deadband so the
        # policy learns to hold or make a change the real servo can execute.
        'servo_deadband_cost_weight': 0.01,
        'healthy_reward': 1.0,
        'vertical_velocity_cost_weight': 0.02,

        # Gait shaping weights. Clearance, unloading, step length, and cadence
        # scales are derived from XML leg length and the current command.
        # Cadence stays fixed at the robot's nominal rhythm.
        'foot_phase_reward_weight': 5.0,
        'stance_foot_height_cost_weight': 0.5,
        'swing_sole_tilt_cost_weight': 0.5,
        'contact_phase_cost_weight': 1.0,
        'flight_cost_weight': 1.0,
        'swing_lift_reward_weight': 2.0,
        'swing_airborne_reward_weight': 2.0,
        'swing_foot_forward_reward_weight': 1.0,
        'foot_slip_cost_weight': 12.0,
        'energy_cost_weight': 0.005,

        # Termination.
        'terminate_when_unhealthy': True,
        'healthy_z_range': (0.034, 0.20),
        'minimum_upright': 0.70,
    })


class Joystick(mjx_env.MjxEnv):
    """Tracks body-forward speed and turn rate with deployment observations."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: dict = None,
        xml_path: str | Path | None = None,
    ):
        # Permit explicit overrides when evaluating a pre-disturbance config,
        # but preserve its undisturbed behavior unless explicitly enabled.
        with config.unlocked():
            for name, value in _YAW_DISTURBANCE_DEFAULTS.items():
                if name not in config:
                    config[name] = (
                        0.0 if name == 'yaw_disturbance_probability' else value
                    )
        # Older command-policy configs keep their sampled command distribution
        # when evaluated with this source. Reward changes still require matching
        # the saved source snapshot for historical return reproduction.
        with config.unlocked():
            additions = {
                **_REWARD_SHAPING_DEFAULTS,
                'command_stage': 'custom',
                'command_forward_velocity_range': config.get('forward_velocity_range', (0.0, config.target_velocity)),
                'command_mode_probabilities': (0.15, 0.15, 0.30, 0.40),
                'command_resample_in_episode': True,
                'command_alternate_speed_endpoints': False,
                'command_speed_levels': (),
            }
            for name, value in additions.items():
                if name not in config:
                    config[name] = value
        super().__init__(config, config_overrides)
        self._configure_yaw_disturbance(config)
        self._configure_reward_shaping(config)

        path = Path(xml_path) if xml_path is not None else consts.task_to_xml(task)
        self._xml_path = path.as_posix()
        self._mj_model = mujoco.MjModel.from_xml_path(path.as_posix())
        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self._mj_model.opt.iterations = 10
        self._mj_model.opt.ls_iterations = 6
        if self._config.impl == 'warp' and not hasattr(mjx.Data, 'where'):
            raise RuntimeError(
                'Rick MJX-Warp requires mujoco-mjx[warp]>=3.16.0. '
                'Use --impl=jax with older installations.'
            )
        model_kwargs = {}
        if self._config.impl == 'warp':
            from mujoco.mjx.warp import types as warp_types
            # Stable staging buffers avoid repeated CUDA graph captures when
            # XLA changes buffer addresses between PPO iterations.
            model_kwargs['graph_mode'] = warp_types.GraphMode.WARP_STAGED
        self._mjx_model = mjx_env.put_model(
            self._mj_model, impl=self._config.impl, **model_kwargs
        )

        self._action_dim = self._mjx_model.nu
        self._history_len = config.command_history_length
        if self._history_len < 2:
            raise ValueError("command_history_length must be at least 2")

        self._target_velocity = config.target_velocity
        self._configure_motion_commands(config)
        self._action_scale = config.action_scale
        self._step_frequency = config.step_frequency
        self._servo_response_range = config.servo_response_range
        self._servo_center_offset_scale_us = (
            config.servo_center_offset_scale_us
        )
        self._servo_gain_range = config.servo_gain_range
        self._servo_voltage_scale_range = (
            config.servo_voltage_scale_range
        )
        self._servo_strength_range = config.servo_strength_range
        self._servo_speed_range = config.servo_speed_range
        self._servo_stiffness_range = config.servo_stiffness_range
        self._servo_frictionloss_range = (
            config.servo_frictionloss_range
        )
        self._servo_deadband_range_us = config.servo_deadband_range_us
        self._servo_backlash_range_us = config.servo_backlash_range_us
        self._action_delay_range = config.action_delay_range
        self._max_action_delay = int(config.action_delay_range[1])
        self._action_noise_scale = config.action_noise_scale
        self._foot_friction_range = config.foot_friction_range
        self._center_of_mass_offset_scale = jp.array(
            config.center_of_mass_offset_scale
        )

        self._root_position_noise_scale = jp.array(
            config.root_position_noise_scale
        )
        self._root_tilt_noise_scale = config.root_tilt_noise_scale
        self._joint_position_noise_scale = (
            config.joint_position_noise_scale
        )
        self._root_linear_velocity_noise_scale = (
            config.root_linear_velocity_noise_scale
        )
        self._root_angular_velocity_noise_scale = (
            config.root_angular_velocity_noise_scale
        )
        self._joint_velocity_noise_scale = (
            config.joint_velocity_noise_scale
        )

        self._accelerometer_noise_scale = (
            config.accelerometer_noise_scale
        )
        self._accelerometer_bias_scale = (
            config.accelerometer_bias_scale
        )
        self._accelerometer_range = config.accelerometer_range
        self._gyro_noise_scale = config.gyro_noise_scale
        self._gyro_bias_scale = config.gyro_bias_scale
        self._gyro_obs_scale = config.gyro_obs_scale
        self._madgwick_beta = config.madgwick_beta

        self._velocity_tracking_weight = config.velocity_tracking_weight
        self._tracking_sigma = config.tracking_sigma
        self._sideways_tracking_sigma = config.get(
            'sideways_tracking_sigma', config.tracking_sigma
        )
        self._sideways_velocity_cost_weight = (
            config.sideways_velocity_cost_weight
        )
        self._yaw_rate_cost_weight = config.yaw_rate_cost_weight
        self._heading_cost_weight = config.heading_cost_weight
        # Missing keys retain old saved checkpoints' observation/reward semantics.
        self._heading_observation = config.get('heading_observation', False)
        self._gyro_bias_calibration = config.get('gyro_bias_calibration', False)
        self._gyro_bias_calibration_samples = int(config.get(
            'gyro_bias_calibration_samples', 100
        ))
        if self._gyro_bias_calibration_samples < 1:
            raise ValueError('gyro_bias_calibration_samples must be positive')
        self._cycle_sideways_velocity_cost_weight = config.get(
            'cycle_sideways_velocity_cost_weight', 0.0
        )
        self._cycle_yaw_rate_cost_weight = config.get(
            'cycle_yaw_rate_cost_weight', 0.0
        )
        if self._step_frequency <= 0:
            raise ValueError('step_frequency must be positive')
        # Fractional oldest sample gives exactly one gait period, including
        # frequencies whose period is not an integer number of control steps.
        self._cycle_samples = 1.0 / (self._step_frequency * self.dt)
        self._cycle_window = int(np.ceil(self._cycle_samples))
        self._cycle_weights = jp.ones(self._cycle_window).at[0].set(
            self._cycle_samples - (self._cycle_window - 1)
        ) / self._cycle_samples
        self._orientation_cost_weight = config.orientation_cost_weight
        self._roll_pitch_rate_cost_weight = (
            config.roll_pitch_rate_cost_weight
        )
        self._action_rate_cost_weight = config.action_rate_cost_weight
        self._action_acceleration_cost_weight = (
            config.action_acceleration_cost_weight
        )
        self._servo_deadband_cost_weight = (
            config.servo_deadband_cost_weight
        )
        self._healthy_reward = config.healthy_reward
        self._vertical_velocity_cost_weight = (
            config.vertical_velocity_cost_weight
        )

        self._foot_phase_reward_weight = config.foot_phase_reward_weight
        self._auto_step_frequency = bool(
            config.get('auto_step_frequency', True)
        )
        self._stance_foot_height_cost_weight = (
            config.stance_foot_height_cost_weight
        )
        self._swing_sole_tilt_cost_weight = (
            config.swing_sole_tilt_cost_weight
        )
        self._contact_phase_cost_weight = config.contact_phase_cost_weight
        self._flight_cost_weight = config.flight_cost_weight
        self._swing_lift_reward_weight = config.swing_lift_reward_weight
        self._swing_airborne_reward_weight = (
            config.swing_airborne_reward_weight
        )
        self._swing_foot_forward_reward_weight = (
            config.swing_foot_forward_reward_weight
        )
        self._foot_slip_cost_weight = config.foot_slip_cost_weight
        self._energy_cost_weight = config.energy_cost_weight

        self._terminate_when_unhealthy = config.terminate_when_unhealthy
        self._healthy_z_range = config.healthy_z_range
        self._minimum_upright = config.minimum_upright

        self._body_idx = mujoco.mj_name2id(
            self._mj_model,
            mujoco.mjtObj.mjOBJ_BODY.value,
            'body',
        )
        self._imu_site_id = mujoco.mj_name2id(
            self._mj_model,
            mujoco.mjtObj.mjOBJ_SITE.value,
            'imu',
        )
        self._imu_body_id = self._mj_model.site_bodyid[self._imu_site_id]
        self._imu_site_quat = self._mjx_model.site_quat[self._imu_site_id]
        self._feet_site_ids = jp.array([
            mujoco.mj_name2id(
                self._mj_model,
                mujoco.mjtObj.mjOBJ_SITE.value,
                'left_foot_center',
            ),
            mujoco.mj_name2id(
                self._mj_model,
                mujoco.mjtObj.mjOBJ_SITE.value,
                'right_foot_center',
            ),
        ])
        self._foot_floor_pair_ids = jp.array([
            mujoco.mj_name2id(
                self._mj_model,
                mujoco.mjtObj.mjOBJ_PAIR.value,
                'left_foot_floor',
            ),
            mujoco.mj_name2id(
                self._mj_model,
                mujoco.mjtObj.mjOBJ_PAIR.value,
                'right_foot_floor',
            ),
        ])
        foot_contact_sensor_ids = [
            mujoco.mj_name2id(
                self._mj_model,
                mujoco.mjtObj.mjOBJ_SENSOR.value,
                'left_foot_floor_found',
            ),
            mujoco.mj_name2id(
                self._mj_model,
                mujoco.mjtObj.mjOBJ_SENSOR.value,
                'right_foot_floor_found',
            ),
        ]
        self._foot_contact_sensor_adrs = jp.array(
            [
                self._mj_model.sensor_adr[sensor_id]
                for sensor_id in foot_contact_sensor_ids
            ]
        )

        # Robot convention: +X sideways, -Y forward, +Z up.
        self._forward_world = jp.array([0.0, -1.0, 0.0])
        self._forward_world_xy = jp.array([0.0, -1.0])

        # The settled pose and the servo targets holding it are stored separately.
        default_keyframe = self._mj_model.key('basic')
        self._default_qpos = jp.array(default_keyframe.qpos)
        self._default_pose = jp.array(default_keyframe.ctrl)
        default_data = mujoco.MjData(self._mj_model)
        mujoco.mj_resetDataKeyframe(
            self._mj_model, default_data, default_keyframe.id
        )
        mujoco.mj_forward(self._mj_model, default_data)
        self._sole_corners = jp.array(self._get_sole_corners(default_data))
        actuator_joint_ids = self._mj_model.actuator_trnid[:, 0]
        # Characteristic physical scales derived from the robot's XML leg length
        # (hip joints / CoM above the standing soles, invariant to CAD body
        # origin) and pendulum dynamics.
        nominal_foot_z = float(
            np.mean(default_data.site_xpos[np.asarray(self._feet_site_ids), 2])
        )
        hip_z = float(np.max(default_data.xanchor[actuator_joint_ids, 2]))
        com_z = float(default_data.subtree_com[self._body_idx, 2])
        leg_length = max(
            hip_z - nominal_foot_z,
            1.5 * (com_z - nominal_foot_z),
            0.05,
        )
        self._natural_clearance_scale = 0.10 * leg_length
        self._natural_contact_depth = 0.04 * leg_length
        if self._auto_step_frequency:
            self._step_frequency = float(
                np.sqrt(9.81 / leg_length) / (2.0 * np.pi)
            )
            self._config['step_frequency'] = self._step_frequency
            self._cycle_samples = 1.0 / (self._step_frequency * self.dt)
            self._cycle_window = int(np.ceil(self._cycle_samples))
            self._cycle_weights = jp.ones(self._cycle_window).at[0].set(
                self._cycle_samples - (self._cycle_window - 1)
            ) / self._cycle_samples
        self._ctrl_min = self._mjx_model.actuator_ctrlrange[:, 0]
        self._ctrl_max = self._mjx_model.actuator_ctrlrange[:, 1]
        self._actuated_dof_ids = jp.array(
            self._mj_model.jnt_dofadr[actuator_joint_ids]
        )

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def reset(self, rng: jp.ndarray) -> mjx_env.State:
        (
            root_position_key,
            root_tilt_key,
            joint_position_key,
            root_linear_velocity_key,
            root_angular_velocity_key,
            joint_velocity_key,
            phase_key,
            accelerometer_bias_key,
            gyro_bias_key,
            servo_center_key,
            servo_gain_key,
            servo_voltage_key,
            servo_strength_key,
            servo_speed_key,
            servo_stiffness_key,
            servo_frictionloss_key,
            servo_response_key,
            servo_deadband_key,
            servo_backlash_key,
            action_delay_key,
            foot_friction_key,
            center_of_mass_key,
            initial_gyro_key,
            step_key,
        ) = jax.random.split(rng, 24)

        qpos = self._default_qpos
        root_position_noise = jax.random.uniform(
            root_position_key,
            (3,),
            minval=-self._root_position_noise_scale,
            maxval=self._root_position_noise_scale,
        )
        qpos = qpos.at[:3].add(root_position_noise)

        roll_pitch = jax.random.uniform(
            root_tilt_key,
            (2,),
            minval=-self._root_tilt_noise_scale,
            maxval=self._root_tilt_noise_scale,
        )
        tilt_quat = self._roll_pitch_quat(roll_pitch)
        root_quat = math.quat_mul(qpos[3:7], tilt_quat)
        root_quat = root_quat / jp.linalg.norm(root_quat)
        qpos = qpos.at[3:7].set(root_quat)
        joint_position_noise = jax.random.uniform(
            joint_position_key,
            (self._action_dim,),
            minval=-self._joint_position_noise_scale,
            maxval=self._joint_position_noise_scale,
        )
        qpos = qpos.at[7:].add(joint_position_noise)

        qvel = jp.zeros((self._mjx_model.nv,))
        qvel = qvel.at[:3].set(jax.random.uniform(
            root_linear_velocity_key,
            (3,),
            minval=-self._root_linear_velocity_noise_scale,
            maxval=self._root_linear_velocity_noise_scale,
        ))
        qvel = qvel.at[3:6].set(jax.random.uniform(
            root_angular_velocity_key,
            (3,),
            minval=-self._root_angular_velocity_noise_scale,
            maxval=self._root_angular_velocity_noise_scale,
        ))
        qvel = qvel.at[6:].set(jax.random.uniform(
            joint_velocity_key,
            (self._action_dim,),
            minval=-self._joint_velocity_noise_scale,
            maxval=self._joint_velocity_noise_scale,
        ))

        servo_center_offset = (
            jax.random.uniform(
                servo_center_key,
                (self._action_dim,),
                minval=-self._servo_center_offset_scale_us,
                maxval=self._servo_center_offset_scale_us,
            )
            / _SERVO_US_PER_RADIAN
        )
        servo_gain = self._sample_uniform(
            servo_gain_key,
            self._servo_gain_range,
            (self._action_dim,),
        )
        servo_voltage_scale = self._sample_uniform(
            servo_voltage_key,
            self._servo_voltage_scale_range,
            (),
        )
        servo_strength = servo_voltage_scale * self._sample_uniform(
            servo_strength_key,
            self._servo_strength_range,
            (self._action_dim,),
        )
        servo_speed = servo_voltage_scale * self._sample_uniform(
            servo_speed_key,
            self._servo_speed_range,
            (self._action_dim,),
        )
        servo_stiffness = self._sample_uniform(
            servo_stiffness_key,
            self._servo_stiffness_range,
            (self._action_dim,),
        )
        servo_frictionloss = self._sample_uniform(
            servo_frictionloss_key,
            self._servo_frictionloss_range,
            (self._action_dim,),
        )
        servo_response = self._sample_uniform(
            servo_response_key,
            self._servo_response_range,
            (self._action_dim,),
        )
        servo_deadband_us = self._sample_uniform(
            servo_deadband_key,
            self._servo_deadband_range_us,
            (self._action_dim,),
        )
        servo_backlash = (
            self._sample_uniform(
                servo_backlash_key,
                self._servo_backlash_range_us,
                (self._action_dim,),
            )
            / _SERVO_US_PER_RADIAN
        )
        action_delay = jax.random.randint(
            action_delay_key,
            (),
            minval=int(self._action_delay_range[0]),
            maxval=int(self._action_delay_range[1]) + 1,
        )
        foot_friction = self._sample_uniform(
            foot_friction_key,
            self._foot_friction_range,
            (),
        )
        center_of_mass_offset = jax.random.uniform(
            center_of_mass_key,
            (3,),
            minval=-self._center_of_mass_offset_scale,
            maxval=self._center_of_mass_offset_scale,
        )

        motor_targets = jp.clip(
            self._default_pose + servo_center_offset,
            self._ctrl_min,
            self._ctrl_max,
        )

        data = mujoco_playground._src.mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=motor_targets,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        reset_model = self._mjx_model.tree_replace({
            'body_ipos': self._body_ipos_with_com_offset(
                center_of_mass_offset
            ),
        })
        data = mjx.forward(reset_model, data)

        phase = jax.random.uniform(
            phase_key,
            minval=-jp.pi,
            maxval=jp.pi,
        )
        command_history = jp.zeros((self._history_len, self._action_dim))
        command_delay_history = jp.zeros(
            (self._max_action_delay + 1, self._action_dim)
        )
        accepted_servo_command = jp.zeros((self._action_dim,))
        servo_action = jp.zeros((self._action_dim,))
        servo_output_target = motor_targets

        accelerometer_bias = jax.random.uniform(
            accelerometer_bias_key,
            (3,),
            minval=-self._accelerometer_bias_scale,
            maxval=self._accelerometer_bias_scale,
        )
        gyro_bias = jax.random.uniform(
            gyro_bias_key,
            (3,),
            minval=-self._gyro_bias_scale,
            maxval=self._gyro_bias_scale,
        )
        gyro_bias_estimate = jp.zeros(3)
        if self._gyro_bias_calibration:
            # Equivalent to averaging N stationary noisy measurements. Only
            # the corrected measurement reaches the policy, never true bias.
            gyro_bias_estimate = gyro_bias + jax.random.normal(
                jax.random.fold_in(gyro_bias_key, 1), (3,)
            ) * self._gyro_noise_scale / np.sqrt(self._gyro_bias_calibration_samples)

        # The Pico runs Madgwick continuously while stopped and preserves the
        # settled estimate on START.  Begin from the simulated IMU site's
        # current attitude to match that prewarmed deployment behavior.
        imu_quat = self._prewarmed_attitude_estimate(
            self._get_imu_quat(data)
        )
        gravity_estimate = self._gravity_from_quat(imu_quat)
        heading_reference = self._heading_from_quat(imu_quat)
        command_key = jax.random.fold_in(rng, 242)
        target_command, command_ticks = self._sample_motion_command(command_key)
        # START always ramps from rest, as on the robot.
        motion_command = jp.zeros(2)
        true_heading_reference = self._heading_from_quat(data.qpos[3:7])
        _, gyro_local = self._get_imu_signals(data)
        gyro_measurement = (
            gyro_local
            + gyro_bias
            - gyro_bias_estimate
            + jax.random.normal(initial_gyro_key, (3,))
            * self._gyro_noise_scale
        )
        observation = self._get_observation(
            command_history=command_history,
            gravity_estimate=gravity_estimate,
            gyro_measurement=gyro_measurement,
            phase=phase,
            heading_error=jp.array(0.0),
            motion_command=motion_command,
        )

        zero = jp.array(0.0)
        metrics = {
            'yaw_disturbance_torque': zero,
            'heading_error': zero,
            'target_velocity': motion_command[0],
            'target_yaw_rate': motion_command[1],
            'cost_standing': zero,
            'reward_velocity_tracking': zero,
            'reward_velocity_tracking_instant': zero,
            'reward_velocity_tracking_cycle': zero,
            'cycle_forward_velocity_error': zero,
            'reward_foot_phase': zero,
            'reward_swing_lift': zero,
            'reward_swing_airborne': zero,
            'reward_swing_foot_forward': zero,
            'reward_alive': zero,
            'cost_sideways_velocity': zero,
            'cost_yaw_rate': zero,
            'cost_cycle_sideways_velocity': zero,
            'cost_cycle_yaw_rate': zero,
            'cycle_sideways_velocity': zero,
            'cycle_yaw_rate': zero,
            'cycle_window_ready': zero,
            'estimated_heading_error': zero,
            'cost_heading': zero,
            'cost_orientation': zero,
            'cost_roll_pitch_rate': zero,
            'cost_action_rate': zero,
            'cost_action_acceleration': zero,
            'cost_servo_deadband': zero,
            'cost_vertical_velocity': zero,
            'cost_foot_slip': zero,
            'cost_energy': zero,
            'x_position': zero,
            'y_position': zero,
            'x_velocity': zero,
            'y_velocity': zero,
            'forward_velocity': zero,
            'sideways_velocity': zero,
            'vertical_velocity': zero,
            'yaw_rate': zero,
            'heading_alignment': zero,
            'uprightness': zero,
            'left_foot_height': zero,
            'right_foot_height': zero,
            'left_toe_clearance': zero,
            'right_toe_clearance': zero,
            'left_heel_clearance': zero,
            'right_heel_clearance': zero,
            'desired_left_foot_height': zero,
            'desired_right_foot_height': zero,
            'left_foot_forward_position': zero,
            'right_foot_forward_position': zero,
            'left_foot_contact': zero,
            'right_foot_contact': zero,
            'foot_friction': foot_friction,
            'center_of_mass_offset_x': center_of_mass_offset[0],
            'center_of_mass_offset_y': center_of_mass_offset[1],
            'center_of_mass_offset_z': center_of_mass_offset[2],
        }

        return mjx_env.State(
            data=data,
            obs=observation,
            reward=zero,
            done=zero,
            metrics=metrics,
            info={
                'command_history': command_history,
                'command_delay_history': command_delay_history,
                'accepted_servo_command': accepted_servo_command,
                'servo_action': servo_action,
                'motor_targets': motor_targets,
                'servo_center_offset': servo_center_offset,
                'servo_gain': servo_gain,
                'servo_voltage_scale': servo_voltage_scale,
                'servo_strength': servo_strength,
                'servo_speed': servo_speed,
                'servo_stiffness': servo_stiffness,
                'servo_frictionloss': servo_frictionloss,
                'servo_response': servo_response,
                'servo_deadband_us': servo_deadband_us,
                'servo_backlash': servo_backlash,
                'servo_output_target': servo_output_target,
                'action_delay': action_delay,
                'foot_friction': foot_friction,
                'center_of_mass_offset': center_of_mass_offset,
                'phase': phase,
                'imu_quat': imu_quat,
                'heading_reference': heading_reference,
                'true_heading_reference': true_heading_reference,
                'motion_command': motion_command,
                'target_command': target_command,
                'command_ticks': command_ticks,
                'yaw_disturbance': self._reset_yaw_disturbance(
                    jax.random.fold_in(rng, 241)
                ),
                'drift_history': jp.zeros((self._cycle_window, 3)),
                'drift_history_count': jp.array(0, dtype=jp.int32),
                'previous_imu_velocity_world': jp.zeros((3,)),
                'accelerometer_bias': accelerometer_bias,
                'gyro_bias': gyro_bias,
                'gyro_bias_estimate': gyro_bias_estimate,
                'rng': step_key,
            },
        )

    def step(
        self,
        state: mjx_env.State,
        action: jp.ndarray,
    ) -> mjx_env.State:
        (
            action_noise_key,
            accelerometer_noise_key,
            gyro_noise_key,
            step_key,
        ) = jax.random.split(
            state.info['rng'], 4
        )

        motion_command, target_command, command_ticks = self._step_motion_command(
            state.info['motion_command'], state.info['target_command'],
            state.info['command_ticks'], jax.random.fold_in(step_key, 242),
        )
        heading_reference = self._wrap_phase(
            state.info['heading_reference'] + motion_command[1] * self.dt
        )
        true_heading_reference = self._wrap_phase(
            state.info['true_heading_reference'] + motion_command[1] * self.dt
        )

        # The controller knows this clipped command, so this is what enters
        # its observation history.  Simulated servo error remains hidden.
        command = jp.clip(action, -1.0, 1.0)
        command_history = state.info['command_history']
        previous_command = command_history[-1]
        action_rate_cost = (
            self._action_rate_cost_weight
            * jp.sum(jp.square(command - previous_command))
        )
        previous_previous_command = command_history[-2]
        action_acceleration = (
            command - 2.0 * previous_command + previous_previous_command
        )
        action_acceleration_cost = (
            self._action_acceleration_cost_weight
            * jp.sum(jp.square(action_acceleration))
        )

        # Convert policy-command changes into the pulse-width changes sent by
        # the Pico firmware.  This smooth bump is zero for a true hold and for
        # a step at or above the deadband, and largest halfway between them.
        # It therefore discourages ineffective fine adjustments without
        # rewarding gratuitously large action changes.
        action_delta_us = (
            jp.abs(command - previous_command)
            * self._action_scale
            * _SERVO_US_PER_RADIAN
        )
        deadband_fraction = jp.clip(
            action_delta_us / state.info['servo_deadband_us'],
            0.0,
            1.0,
        )
        servo_deadband_cost = (
            self._servo_deadband_cost_weight
            * jp.sum(
                4.0
                * deadband_fraction
                * (1.0 - deadband_fraction)
            )
        )

        # Delay the hidden actuator path while retaining the current sent
        # command in the policy-visible history, exactly as on the Pico.
        command_delay_history = jp.roll(
            state.info['command_delay_history'], shift=-1, axis=0
        )
        command_delay_history = command_delay_history.at[-1].set(command)
        delay_index = self._max_action_delay - state.info['action_delay']
        delayed_command = jax.lax.dynamic_index_in_dim(
            command_delay_history,
            delay_index,
            axis=0,
            keepdims=False,
        )

        actuation_error = (
            jax.random.normal(action_noise_key, command.shape)
            * self._action_noise_scale
        )
        uncertain_action = jp.clip(
            delayed_command + actuation_error,
            -1.0,
            1.0,
        )

        # MG90S command deadband: retain the last internally accepted target
        # until the pulse-width change is large enough for that servo.
        accepted_servo_command = state.info['accepted_servo_command']
        servo_input_delta_us = (
            jp.abs(uncertain_action - accepted_servo_command)
            * self._action_scale
            * _SERVO_US_PER_RADIAN
        )
        accepted_servo_command = jp.where(
            servo_input_delta_us >= state.info['servo_deadband_us'],
            uncertain_action,
            accepted_servo_command,
        )
        servo_action = (
            state.info['servo_response'] * accepted_servo_command
            + (1.0 - state.info['servo_response'])
            * state.info['servo_action']
        )

        desired_motor_targets = (
            self._default_pose
            + state.info['servo_center_offset']
            + self._action_scale * state.info['servo_gain'] * servo_action
        )
        desired_motor_targets = jp.clip(
            desired_motor_targets,
            self._ctrl_min,
            self._ctrl_max,
        )
        max_target_delta = state.info['servo_speed'] * self.dt
        motor_targets = state.info['motor_targets'] + jp.clip(
            desired_motor_targets - state.info['motor_targets'],
            -max_target_delta,
            max_target_delta,
        )
        # Stateful output-side play.  Small changes and direction reversals are
        # first absorbed by gear lash instead of moving the simulated joint.
        servo_output_target = self._apply_servo_backlash(
            state.info['servo_output_target'],
            motor_targets,
            state.info['servo_backlash'],
        )

        new_command_history = jp.roll(command_history, shift=-1, axis=0)
        new_command_history = new_command_history.at[-1].set(command)

        disturbance, yaw_torque = self._step_yaw_disturbance(
            state.info['yaw_disturbance']
        )
        # xfrc_applied is [world force, world torque]. Clear the previous
        # pulse every step; never rotate qpos or reset the IMU/reference.
        data0 = state.data.replace(xfrc_applied=jp.zeros_like(
            state.data.xfrc_applied
        ).at[self._body_idx, 5].set(yaw_torque))
        stiffness = state.info['servo_stiffness']
        actuator_gainprm = self._mjx_model.actuator_gainprm.at[:, 0].set(
            self._mjx_model.actuator_gainprm[:, 0] * stiffness
        )
        actuator_biasprm = self._mjx_model.actuator_biasprm.at[:, 1].set(
            self._mjx_model.actuator_biasprm[:, 1] * stiffness
        )
        # Scale kv with sqrt(kp) to approximately preserve damping ratio.
        actuator_biasprm = actuator_biasprm.at[:, 2].set(
            self._mjx_model.actuator_biasprm[:, 2]
            * jp.sqrt(stiffness)
        )
        dof_frictionloss = self._mjx_model.dof_frictionloss.at[
            self._actuated_dof_ids
        ].set(state.info['servo_frictionloss'])
        pair_friction = self._mjx_model.pair_friction.at[
            self._foot_floor_pair_ids, :2
        ].set(jp.full(
            (self._foot_floor_pair_ids.shape[0], 2),
            state.info['foot_friction'],
        ))
        body_ipos = self._body_ipos_with_com_offset(
            state.info['center_of_mass_offset']
        )
        step_model = self._mjx_model.tree_replace({
            'actuator_gainprm': actuator_gainprm,
            'actuator_biasprm': actuator_biasprm,
            'actuator_forcerange': (
                self._mjx_model.actuator_forcerange
                * state.info['servo_strength'][:, None]
            ),
            'dof_frictionloss': dof_frictionloss,
            'pair_friction': pair_friction,
            'body_ipos': body_ipos,
        })
        data = mjx_env.step(
            step_model,
            data0,
            servo_output_target,
            self.n_substeps,
        )

        phase = self._wrap_phase(
            state.info['phase']
            + 2.0 * jp.pi * self._step_frequency * self.dt
        )

        root_quat = data.qpos[3:7]
        root_quat = root_quat / jp.linalg.norm(root_quat)
        inv_quat = self._quat_inverse(root_quat)
        gravity_local = math.rotate(
            jp.array([0.0, 0.0, -1.0]),
            inv_quat,
        )
        uprightness = -gravity_local[2]
        body_forward_world = math.rotate(self._forward_world, root_quat)
        body_forward_xy = body_forward_world[:2]
        body_forward_xy = body_forward_xy / jp.maximum(
            jp.linalg.norm(body_forward_xy),
            1e-6,
        )

        root_angular_velocity_local = data.qvel[3:6]
        root_angular_velocity_world = math.rotate(
            root_angular_velocity_local,
            root_quat,
        )
        yaw_rate = root_angular_velocity_world[2]

        com_before = data0.subtree_com[self._body_idx]
        com_after = data.subtree_com[self._body_idx]
        velocity_world = (com_after - com_before) / self.dt
        forward_velocity, sideways_velocity = self._body_planar_velocity(
            velocity_world, body_forward_xy
        )

        # Store errors against each sample's own command, so changing targets
        # do not compare past motion with only the most recent target.
        forward_velocity_error = forward_velocity - motion_command[0]
        yaw_rate_error = yaw_rate - motion_command[1]
        drift_history, drift_count, cycle_velocity, cycle_ready = (
            self._update_cycle_velocity(
                state.info['drift_history'], state.info['drift_history_count'],
                jp.array([sideways_velocity, yaw_rate_error, forward_velocity_error]),
            )
        )
        velocity_tracking_reward, instant_tracking_reward, cycle_tracking_reward = (
            self._get_velocity_tracking_reward(
                forward_velocity_error, cycle_velocity[2], cycle_ready
            )
        )
        sideways_velocity_cost = (
            self._sideways_velocity_cost_weight
            * jp.square(sideways_velocity)
            / self._sideways_tracking_sigma
        )

        desired_foot_heights = self._desired_foot_heights(phase)
        foot_positions_before = data0.site_xpos[self._feet_site_ids]
        foot_positions_after = data.site_xpos[self._feet_site_ids]
        toe_clearance, heel_clearance = self._get_foot_clearances(data)
        foot_heights = jp.minimum(toe_clearance, heel_clearance)

        foot_velocities_world = (
            foot_positions_after - foot_positions_before
        ) / self.dt

        foot_contacts = (
            data.sensordata[self._foot_contact_sensor_adrs] > 0
        )
        swing_foot_forward_reward, foot_forward_positions = (
            self._get_swing_foot_forward_reward(
                foot_positions_after,
                com_after,
                body_forward_xy,
                phase,
                foot_velocities_world,
                velocity_world,
                motion_command,
                yaw_rate,
                foot_contacts,
                toe_clearance,
                heel_clearance,
            )
        )
        foot_phase_reward, swing_lift_reward, swing_airborne_reward = (
            self._get_foot_height_rewards(
                phase, foot_contacts, toe_clearance, heel_clearance
            )
        )
        foot_contacts_before = (
            data0.sensordata[self._foot_contact_sensor_adrs] > 0
        )
        activity = self._walking_activity(motion_command)
        # Blend stepping into quiet, supported standing at zero command.
        foot_phase_reward *= activity
        swing_lift_reward *= activity
        swing_airborne_reward *= activity
        swing_foot_forward_reward *= activity
        standing_cost = (1.0 - activity) * self._standing_reward_weight * (
            jp.sum(~foot_contacts)
            + jp.sum(jp.square(foot_heights / self._natural_clearance_scale))
        )
        foot_slip_cost = self._get_foot_slip_cost(
            foot_velocities_world,
            foot_contacts_before,
            foot_contacts,
            phase,
            toe_clearance,
            heel_clearance,
        )

        # These global quantities shape training but never enter observation.
        yaw_rate_cost = self._yaw_rate_cost_weight * jp.square(yaw_rate_error)
        cycle_sideways_velocity_cost = (
            self._cycle_sideways_velocity_cost_weight
            * jp.square(cycle_velocity[0]) / self._sideways_tracking_sigma
            * cycle_ready
        )
        cycle_yaw_rate_cost = (
            self._cycle_yaw_rate_cost_weight * jp.square(cycle_velocity[1])
            * cycle_ready
        )
        roll_pitch_rate_cost = (
            self._roll_pitch_rate_cost_weight
            * jp.sum(jp.square(root_angular_velocity_local[:2]))
        )

        heading_alignment = jp.dot(
            body_forward_xy,
            jp.array([jp.sin(true_heading_reference), -jp.cos(true_heading_reference)]),
        )
        heading_cost = (
            self._heading_cost_weight * (1.0 - heading_alignment)
        )

        tilt_cost = (
            self._orientation_cost_weight
            * jp.sum(jp.square(gravity_local[:2]))
        )
        vertical_velocity_cost = (
            self._vertical_velocity_cost_weight
            * jp.square(velocity_world[2])
        )
        base_height = data.subtree_com[self._body_idx, 2]
        energy_cost = (
            self._energy_cost_weight
            * jp.sum(jp.abs(data.actuator_force * data.qvel[6:]))
        )

        min_z, max_z = self._healthy_z_range
        healthy_height = (base_height >= min_z) & (base_height <= max_z)
        healthy_orientation = uprightness >= self._minimum_upright
        is_healthy = healthy_height & healthy_orientation
        is_healthy_float = is_healthy.astype(jp.float32)
        done = (
            1.0 - is_healthy_float
            if self._terminate_when_unhealthy
            else jp.array(0.0)
        )
        healthy_reward = (
            self._healthy_reward * is_healthy_float
            - self._cycle_samples * self._healthy_reward * done
        )

        reward = (
            velocity_tracking_reward
            + foot_phase_reward
            + swing_lift_reward
            + swing_airborne_reward
            + swing_foot_forward_reward
            + healthy_reward
            - sideways_velocity_cost
            - yaw_rate_cost
            - cycle_sideways_velocity_cost
            - cycle_yaw_rate_cost
            - heading_cost
            - standing_cost
            - tilt_cost
            - roll_pitch_rate_cost
            - action_rate_cost
            - action_acceleration_cost
            - servo_deadband_cost
            - vertical_velocity_cost
            - foot_slip_cost
            - energy_cost
        )

        # Build deployment-like raw IMU samples at the XML's IMU site.  An
        # accelerometer measures specific force (linear acceleration minus
        # gravity), not orientation.  Its offset from the COM means angular
        # motion also contributes to the simulated measurement.
        imu_velocity_world = self._get_imu_linear_velocity_world(data0, data)
        linear_acceleration_world = (
            imu_velocity_world - state.info['previous_imu_velocity_world']
        ) / self.dt
        specific_force_world = (
            linear_acceleration_world - jp.array([0.0, 0.0, -9.81])
        ) / 9.81
        specific_force_local = (
            data.site_xmat[self._imu_site_id].T @ specific_force_world
        )
        accelerometer_measurement = jp.clip(
            specific_force_local
            + state.info['accelerometer_bias']
            + jax.random.normal(accelerometer_noise_key, (3,))
            * self._accelerometer_noise_scale,
            -self._accelerometer_range,
            self._accelerometer_range,
        )
        _, gyro_local = self._get_imu_signals(data)
        gyro_measurement = (
            gyro_local
            + state.info['gyro_bias']
            - state.info['gyro_bias_estimate']
            + jax.random.normal(gyro_noise_key, (3,))
            * self._gyro_noise_scale
        )
        imu_quat = self._update_attitude_estimate(
            state.info['imu_quat'],
            gyro_measurement,
            accelerometer_measurement,
        )
        gravity_estimate = self._gravity_from_quat(imu_quat)
        heading_error = self._wrap_phase(
            self._heading_from_quat(imu_quat) - heading_reference
        )

        # Heading uses the same noisy filter as the Pico, not privileged root
        # orientation. All reward state above remains training-only.
        observation = self._get_observation(
            command_history=new_command_history,
            gravity_estimate=gravity_estimate,
            gyro_measurement=gyro_measurement,
            phase=phase,
            heading_error=heading_error,
            motion_command=motion_command,
        )

        state.metrics.update(
            yaw_disturbance_torque=yaw_torque,
            heading_error=self._wrap_phase(
                jp.arctan2(body_forward_xy[0], -body_forward_xy[1]) - true_heading_reference
            ),
            target_velocity=motion_command[0],
            target_yaw_rate=motion_command[1],
            cost_standing=-standing_cost,
            reward_velocity_tracking=velocity_tracking_reward,
            reward_velocity_tracking_instant=instant_tracking_reward,
            reward_velocity_tracking_cycle=cycle_tracking_reward,
            cycle_forward_velocity_error=cycle_velocity[2],
            reward_foot_phase=foot_phase_reward,
            reward_swing_lift=swing_lift_reward,
            reward_swing_airborne=swing_airborne_reward,
            reward_swing_foot_forward=swing_foot_forward_reward,
            reward_alive=healthy_reward,
            cost_sideways_velocity=-sideways_velocity_cost,
            cost_yaw_rate=-yaw_rate_cost,
            cost_cycle_sideways_velocity=-cycle_sideways_velocity_cost,
            cost_cycle_yaw_rate=-cycle_yaw_rate_cost,
            cycle_sideways_velocity=cycle_velocity[0],
            cycle_yaw_rate=cycle_velocity[1],
            cycle_window_ready=cycle_ready.astype(jp.float32),
            estimated_heading_error=heading_error,
            cost_heading=-heading_cost,
            cost_orientation=-tilt_cost,
            cost_roll_pitch_rate=-roll_pitch_rate_cost,
            cost_action_rate=-action_rate_cost,
            cost_action_acceleration=-action_acceleration_cost,
            cost_servo_deadband=-servo_deadband_cost,
            cost_vertical_velocity=-vertical_velocity_cost,
            cost_foot_slip=-foot_slip_cost,
            cost_energy=-energy_cost,
            x_position=data.qpos[0],
            y_position=data.qpos[1],
            x_velocity=velocity_world[0],
            y_velocity=velocity_world[1],
            forward_velocity=forward_velocity,
            sideways_velocity=sideways_velocity,
            vertical_velocity=velocity_world[2],
            yaw_rate=yaw_rate,
            heading_alignment=heading_alignment,
            uprightness=uprightness,
            left_foot_height=foot_heights[0],
            right_foot_height=foot_heights[1],
            left_toe_clearance=toe_clearance[0],
            right_toe_clearance=toe_clearance[1],
            left_heel_clearance=heel_clearance[0],
            right_heel_clearance=heel_clearance[1],
            desired_left_foot_height=desired_foot_heights[0],
            desired_right_foot_height=desired_foot_heights[1],
            left_foot_forward_position=foot_forward_positions[0],
            right_foot_forward_position=foot_forward_positions[1],
            left_foot_contact=foot_contacts[0].astype(jp.float32),
            right_foot_contact=foot_contacts[1].astype(jp.float32),
            foot_friction=state.info['foot_friction'],
            center_of_mass_offset_x=(
                state.info['center_of_mass_offset'][0]
            ),
            center_of_mass_offset_y=(
                state.info['center_of_mass_offset'][1]
            ),
            center_of_mass_offset_z=(
                state.info['center_of_mass_offset'][2]
            ),
        )

        return state.replace(
            data=data,
            obs=observation,
            reward=reward,
            done=done,
            info={
                **state.info,
                'command_history': new_command_history,
                'command_delay_history': command_delay_history,
                'accepted_servo_command': accepted_servo_command,
                'servo_action': servo_action,
                'motor_targets': motor_targets,
                'servo_output_target': servo_output_target,
                'phase': phase,
                'imu_quat': imu_quat,
                'heading_reference': heading_reference,
                'true_heading_reference': true_heading_reference,
                'motion_command': motion_command,
                'target_command': target_command,
                'command_ticks': command_ticks,
                'drift_history': drift_history,
                'drift_history_count': drift_count,
                'previous_imu_velocity_world': imu_velocity_world,
                'rng': step_key,
                'yaw_disturbance': disturbance,
            },
        )

    def _configure_yaw_disturbance(self, config):
        self._yaw_probability = float(config.yaw_disturbance_probability)
        if not np.isfinite(self._yaw_probability) or not 0 <= self._yaw_probability <= 1:
            raise ValueError('yaw_disturbance_probability must be in [0, 1]')
        self._yaw_torque_range = tuple(config.yaw_disturbance_torque_range)
        self._yaw_start_range = tuple(config.yaw_disturbance_start_range)
        self._yaw_interval_range = tuple(config.yaw_disturbance_interval_range)
        for name, bounds in (
            ('torque', self._yaw_torque_range),
            ('start', self._yaw_start_range),
            ('interval', self._yaw_interval_range),
        ):
            if (len(bounds) != 2 or not np.all(np.isfinite(bounds))
                    or bounds[0] > bounds[1]):
                raise ValueError(f'yaw_disturbance_{name}_range must be finite and ordered')
            if name != 'torque' and bounds[0] < self.dt:
                raise ValueError(f'yaw_disturbance_{name}_range must be >= ctrl_dt')
        duration = float(config.yaw_disturbance_duration)
        self._yaw_ramp_seconds = float(config.yaw_disturbance_ramp_seconds)
        if not np.isfinite(duration) or duration < self.dt:
            raise ValueError('yaw_disturbance_duration must be finite and >= ctrl_dt')
        if not np.isfinite(self._yaw_ramp_seconds) or self._yaw_ramp_seconds < 0:
            raise ValueError('yaw_disturbance_ramp_seconds must be finite and >= 0')
        self._yaw_duration_steps = int(np.ceil(duration / self.dt))

    def _yaw_wait_steps(self, key, bounds):
        return jp.ceil(self._sample_uniform(key, bounds, ()) / self.dt).astype(jp.int32)

    def _reset_yaw_disturbance(self, key):
        enable_key, wait_key, key = jax.random.split(key, 3)
        return {
            'enabled': jax.random.bernoulli(enable_key, self._yaw_probability),
            'wait': self._yaw_wait_steps(wait_key, self._yaw_start_range),
            'remaining': jp.array(0, dtype=jp.int32),
            'elapsed': jp.array(0, dtype=jp.int32),
            'torque': jp.array(0.0),
            'rng': key,
        }

    def _step_yaw_disturbance(self, disturbance):
        torque_key, wait_key, key = jax.random.split(disturbance['rng'], 3)
        start = (disturbance['enabled'] & (disturbance['wait'] == 0)
                 & (disturbance['remaining'] == 0))
        scale = (0.25 + 0.75 * jp.minimum(
            disturbance['elapsed'] * self.dt / self._yaw_ramp_seconds, 1.0
        )) if self._yaw_ramp_seconds > 0 else 1.0
        torque = jp.where(start, scale * self._sample_uniform(
            torque_key, self._yaw_torque_range, ()
        ), disturbance['torque'])
        remaining = jp.where(start, self._yaw_duration_steps, disturbance['remaining'])
        active = disturbance['enabled'] & (remaining > 0)
        wait = jp.where(start, self._yaw_wait_steps(
            wait_key, self._yaw_interval_range
        ), disturbance['wait'])
        return {
            **disturbance,
            'wait': jp.where(active, wait, jp.maximum(wait - 1, 0)),
            'remaining': jp.maximum(remaining - 1, 0),
            'elapsed': disturbance['elapsed'] + 1,
            'torque': torque,
            'rng': key,
        }, jp.where(active, torque, 0.0)

    def _get_sole_corners(self, data: mujoco.MjData) -> np.ndarray:
        """Bounds Rick's rectangular soles in their foot-center site frames.

        Use the loaded collision geometry, including MuJoCo's mesh transform,
        rather than dimensions from a different revision of the robot.  The
        first two corners are the toe (-Y), the last two the heel (+Y).
        """
        corners = []
        for side, site_id in zip(('left', 'right'), self._feet_site_ids):
            site_id = int(site_id)
            geom = self._mj_model.geom(f'{side}_foot_collision')
            geom_id = geom.id
            if (
                self._mj_model.geom_bodyid[geom_id]
                != self._mj_model.site_bodyid[site_id]
            ):
                raise ValueError(
                    'Foot collision and center site must share a body'
                )
            if geom.type[0] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = int(geom.dataid[0])
                start = self._mj_model.mesh_vertadr[mesh_id]
                count = self._mj_model.mesh_vertnum[mesh_id]
                vertices = self._mj_model.mesh_vert[start:start + count]
            elif geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX:
                signs = np.indices((2, 2, 2)).reshape(3, -1).T * 2 - 1
                vertices = signs * geom.size
            else:
                raise ValueError('Rick foot collision must be a mesh or box')
            world = (
                vertices @ data.geom_xmat[geom_id].reshape(3, 3).T
                + data.geom_xpos[geom_id]
            )
            local = (
                (world - data.site_xpos[site_id])
                @ data.site_xmat[site_id].reshape(3, 3)
            )
            low, high = local.min(axis=0), local.max(axis=0)
            corners.append([
                [low[0], low[1], low[2]],
                [high[0], low[1], low[2]],
                [low[0], high[1], low[2]],
                [high[0], high[1], low[2]],
            ])
        return np.asarray(corners)

    def _get_foot_clearances(
        self, data: mjx.Data
    ) -> tuple[jp.ndarray, jp.ndarray]:
        """Lowest toe/heel corner heights above this task's flat z=0 floor.

        Keeping both lateral corners catches roll as well as pitch.  Preserve
        negative heights in diagnostics so penetration is visible.
        """
        rotations = data.site_xmat[self._feet_site_ids].reshape((-1, 3, 3))
        heights = (
            jp.einsum('fkj,fj->fk', self._sole_corners, rotations[:, 2, :])
            + data.site_xpos[self._feet_site_ids, 2, None]
        )
        return jp.min(heights[:, :2], axis=1), jp.min(heights[:, 2:], axis=1)

    def _get_foot_height_rewards(
        self,
        phase: jp.ndarray,
        contacts: jp.ndarray,
        toe_clearance: jp.ndarray,
        heel_clearance: jp.ndarray,
    ) -> tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
        """Target-free phase-weighted unloading and clearance rewards."""
        phase_sin = jp.sin(phase + jp.array([0.0, jp.pi]))
        swing = jp.square(jp.maximum(phase_sin, 0.0))
        stance = jp.square(jp.minimum(phase_sin, 0.0))
        clearance = jp.minimum(toe_clearance, heel_clearance)
        c0 = self._natural_clearance_scale
        d0 = self._natural_contact_depth
        opposite_contact = contacts[::-1].astype(clearance.dtype)
        contact = contacts.astype(clearance.dtype)

        # C1-continuous transition at liftoff (clearance = 0): linear unloading
        # slope 1/c0 below 0 joins 1 - exp(-clearance / c0) above 0.
        airborne_clearance = 1.0 - jp.exp(-jp.maximum(clearance, 0.0) / c0)
        swing_credit = (
            jp.minimum(clearance / c0, 0.0)
            + airborne_clearance * opposite_contact
        )
        foot_phase_reward = (
            self._foot_phase_reward_weight * jp.sum(swing * swing_credit)
            - self._stance_foot_height_cost_weight
            * jp.sum(stance * airborne_clearance)
        )
        foot_phase_reward -= self._contact_phase_cost_weight * jp.sum(
            swing * contact + stance * (1.0 - contact)
        )
        sole_tilt = jp.abs(toe_clearance - heel_clearance) / c0
        foot_phase_reward -= self._swing_sole_tilt_cost_weight * jp.sum(
            swing * jp.tanh(sole_tilt)
        )
        foot_phase_reward -= self._flight_cost_weight * (~jp.any(contacts))

        lift_progress = 1.0 - jp.exp(
            -jp.maximum(clearance + d0, 0.0) / (c0 + d0)
        )
        swing_lift_reward = self._swing_lift_reward_weight * jp.sum(
            swing * lift_progress
        )
        swing_airborne_reward = self._swing_airborne_reward_weight * jp.sum(
            swing * (1.0 - contact) * airborne_clearance
        )
        return foot_phase_reward, swing_lift_reward, swing_airborne_reward

    def _get_foot_slip_cost(
        self,
        foot_velocities_world: jp.ndarray,
        foot_contacts_before: jp.ndarray,
        foot_contacts_after: jp.ndarray,
        phase: jp.ndarray,
        toe_clearance: jp.ndarray,
        heel_clearance: jp.ndarray,
    ) -> jp.ndarray:
        """Penalizes stance sliding and mid-swing near-ground scuffing."""
        contacts = (foot_contacts_before & foot_contacts_after).astype(
            foot_velocities_world.dtype
        )
        tangential_speed_squared = jp.sum(
            jp.square(foot_velocities_world[:, :2]),
            axis=1,
        )
        c0 = self._natural_clearance_scale
        swing = jp.square(jp.maximum(
            jp.sin(phase + jp.array([0.0, jp.pi])), 0.0
        ))
        clearance = jp.minimum(toe_clearance, heel_clearance)
        near_ground = jp.exp(-jp.maximum(clearance, 0.0) / (0.5 * c0))
        return self._foot_slip_cost_weight * jp.sum(
            (contacts + 0.5 * swing * (1.0 - contacts) * near_ground)
            * tangential_speed_squared
        )

    def _body_ipos_with_com_offset(
        self,
        center_of_mass_offset: jp.ndarray,
    ) -> jp.ndarray:
        """Applies a shared COM offset without modifying the world body."""
        return self._mjx_model.body_ipos.at[1:].add(
            center_of_mass_offset
        )

    def _configure_reward_shaping(self, config):
        for name in ('tracking_sigma', 'instant_velocity_tracking_sigma',
                     'swing_progress_clearance'):
            if not np.isfinite(config[name]) or config[name] <= 0:
                raise ValueError(f'{name} must be positive and finite')
        fraction = config.cycle_velocity_tracking_fraction
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('cycle_velocity_tracking_fraction must be in [0, 1]')
        self._cycle_velocity_tracking_fraction = fraction
        self._instant_velocity_tracking_sigma = config.instant_velocity_tracking_sigma
        self._swing_progress_clearance = config.swing_progress_clearance

    def _get_velocity_tracking_reward(self, error, cycle_error, cycle_ready):
        # Use only the broad instantaneous term until a full period is recorded.
        # This avoids granting full cycle credit for zero-filled history.
        fraction = jp.where(cycle_ready, self._cycle_velocity_tracking_fraction, 0.0)
        instant = self._velocity_tracking_weight * (1.0 - fraction) * jp.exp(
            -jp.square(error) / self._instant_velocity_tracking_sigma
        )
        cycle = self._velocity_tracking_weight * fraction * jp.exp(
            -jp.square(cycle_error) / self._tracking_sigma
        )
        return instant + cycle, instant, cycle

    def _configure_motion_commands(self, config):
        if not config.heading_observation:
            raise ValueError('motion commands require heading_observation=True')
        if config.get('motion_command_version') != 1:
            raise ValueError('Rick command training requires motion_command_version=1; start a new run')
        for name in ('forward_velocity_range', 'command_forward_velocity_range', 'yaw_rate_range',
                     'command_resampling_time_range'):
            bounds = np.asarray(config[name], dtype=float)
            if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
                raise ValueError(f'{name} must be a finite ordered pair')
        if config.command_resampling_time_range[0] < self.dt:
            raise ValueError('command_resampling_time_range must be at least ctrl_dt')
        for name in ('linear_acceleration_limit', 'yaw_acceleration_limit'):
            if not np.isfinite(config[name]) or config[name] <= 0:
                raise ValueError(f'{name} must be positive and finite')
        self._command_min = jp.array([config.command_forward_velocity_range[0], config.yaw_rate_range[0]])
        self._command_max = jp.array([config.command_forward_velocity_range[1], config.yaw_rate_range[1]])
        if not (config.forward_velocity_range[0] <= 0 <= config.forward_velocity_range[1]
                and config.yaw_rate_range[0] <= 0 <= config.yaw_rate_range[1]):
            raise ValueError('motion command ranges must include zero for standing')
        if not (config.forward_velocity_range[0] <= config.command_forward_velocity_range[0]
                <= config.command_forward_velocity_range[1] <= config.forward_velocity_range[1]):
            raise ValueError('sampled forward commands must fit forward_velocity_range')
        probabilities = np.asarray(config.command_mode_probabilities, dtype=float)
        if (probabilities.shape != (4,) or not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0)):
            raise ValueError('command_mode_probabilities must be four nonnegative probabilities summing to one')
        self._command_mode_thresholds = jp.array(np.cumsum(probabilities))
        if not isinstance(config.command_resample_in_episode, bool):
            raise ValueError('command_resample_in_episode must be a boolean')
        self._command_resample_in_episode = config.command_resample_in_episode
        alternate = config.get('command_alternate_speed_endpoints', False)
        if not isinstance(alternate, bool):
            raise ValueError('command_alternate_speed_endpoints must be a boolean')
        if alternate and not (
                config.command_resample_in_episode
                and 0 < config.command_forward_velocity_range[0] < config.command_forward_velocity_range[1]
                and tuple(config.yaw_rate_range) == (0.0, 0.0)
                and np.allclose(probabilities, [0.0, 0.0, 1.0, 0.0])):
            raise ValueError('Alternating speed endpoints require positive distinct speeds, straight-only modes, zero yaw, and resampling')
        self._command_alternate_speed_endpoints = alternate
        levels = np.asarray(config.get('command_speed_levels', ()), dtype=float)
        if levels.ndim != 1:
            raise ValueError('command_speed_levels must be a one-dimensional sequence')
        if levels.size and not (
                levels.size >= 2 and np.all(np.isfinite(levels))
                and np.all(levels > 0) and np.all(np.diff(levels) > 0)
                and config.command_forward_velocity_range[0] <= levels[0]
                and levels[-1] <= config.command_forward_velocity_range[1]
                and config.command_resample_in_episode and not alternate
                and tuple(config.yaw_rate_range) == (0.0, 0.0)
                and np.allclose(probabilities, [0.0, 0.0, 1.0, 0.0])):
            raise ValueError('Discrete speed levels require ordered positive distinct speeds in range, straight-only modes, zero yaw, resampling, and no endpoint alternation')
        self._command_speed_levels = jp.asarray(levels, dtype=jp.float32)
        self._command_time_range = tuple(config.command_resampling_time_range)
        self._command_acceleration = jp.array([
            config.linear_acceleration_limit, config.yaw_acceleration_limit,
        ])
        if not np.isfinite(config.standing_reward_weight) or config.standing_reward_weight < 0:
            raise ValueError('standing_reward_weight must be nonnegative and finite')
        if not config.forward_velocity_range[0] <= config.target_velocity <= config.forward_velocity_range[1]:
            raise ValueError('target_velocity must be within forward_velocity_range')
        self._standing_reward_weight = config.standing_reward_weight

    def _sample_motion_command(self, rng):
        value_key, mode_key, time_key = jax.random.split(rng, 3)
        command = jax.random.uniform(
            value_key, (2,), minval=self._command_min, maxval=self._command_max
        )
        if self._command_alternate_speed_endpoints:
            first_speed = jp.where(jax.random.bernoulli(value_key),
                                   self._command_max[0], self._command_min[0])
            command = command.at[0].set(first_speed)
        if self._command_speed_levels.size:
            index = jax.random.randint(value_key, (), 0, self._command_speed_levels.size)
            command = command.at[0].set(self._command_speed_levels[index])
        mode = jax.random.uniform(mode_key)
        standing_end, pivot_end, straight_end = self._command_mode_thresholds[:3]
        command = jp.where(mode < pivot_end, command.at[0].set(0.0), command)
        command = jp.where((mode >= pivot_end) & (mode < straight_end), command.at[1].set(0.0), command)
        command = jp.where(mode < standing_end, jp.zeros(2), command)
        duration = jax.random.uniform(time_key, (),
            minval=self._command_time_range[0], maxval=self._command_time_range[1])
        ticks = jp.maximum(jp.ceil(duration / self.dt).astype(jp.int32), 1)
        return command, ticks

    def _step_motion_command(self, command, target, ticks, rng):
        sampled_target, sampled_ticks = self._sample_motion_command(rng)
        if self._command_alternate_speed_endpoints:
            midpoint = 0.5 * (self._command_min[0] + self._command_max[0])
            next_speed = jp.where(target[0] <= midpoint,
                                  self._command_max[0], self._command_min[0])
            sampled_target = jp.array([next_speed, 0.0])
        if self._command_speed_levels.size:
            index = jp.argmin(jp.abs(self._command_speed_levels - target[0]))
            next_speed = self._command_speed_levels[(index + 1) % self._command_speed_levels.size]
            sampled_target = jp.array([next_speed, 0.0])
        resample = (ticks <= 0) & self._command_resample_in_episode
        target = jp.where(resample, sampled_target, target)
        ticks = jp.where(resample, sampled_ticks, ticks)
        limit = self._command_acceleration * self.dt
        command = command + jp.clip(target - command, -limit, limit)
        return command, target, jp.maximum(ticks - 1, 0)

    @staticmethod
    def _walking_activity(command):
        return jp.clip(jp.maximum(jp.abs(command[0]) / 0.01,
                                  jp.abs(command[1]) / 0.10), 0.0, 1.0)

    @staticmethod
    def _body_planar_velocity(velocity_world, body_forward_xy):
        right = jp.array([body_forward_xy[1], -body_forward_xy[0]])
        return velocity_world[:2] @ body_forward_xy, velocity_world[:2] @ right

    def _get_observation(
        self,
        command_history: jp.ndarray,
        gravity_estimate: jp.ndarray,
        gyro_measurement: jp.ndarray,
        phase: jp.ndarray,
        heading_error: jp.ndarray = jp.array(0.0),
        motion_command: jp.ndarray = jp.zeros(2),
    ) -> jp.ndarray:
        """Builds the deployment observation (108 values by default).

        With the default 12-frame history, the layout is:
          0:96   normalized command history, oldest to newest
          96:99  unit gravity estimate in IMU coordinates
          99:102 scaled gyroscope in IMU coordinates
          102:104 sin/cos gait clock
          104:106 commanded body-forward speed (m/s), turn rate (rad/s)
          106:108 sin/cos estimated heading minus commanded heading (optional)
        """
        clock = jp.array([jp.sin(phase), jp.cos(phase)])
        parts = [
            command_history.flatten(),
            gravity_estimate,
            self._gyro_obs_scale * gyro_measurement,
            clock,
            motion_command,
        ]
        if self._heading_observation:
            parts.append(jp.array([jp.sin(heading_error), jp.cos(heading_error)]))
        return jp.clip(
            jp.concatenate(parts),
            -10.0,
            10.0,
        )

    def _update_cycle_velocity(self, history, count, velocity):
        """Sliding full-period mean; no partial-window startup penalty."""
        history = jp.concatenate([history[1:], velocity[None]], axis=0)
        count = jp.minimum(count + 1, self._cycle_window)
        ready = count == self._cycle_window
        mean = self._cycle_weights @ history
        return history, count, jp.where(ready, mean, jp.zeros_like(velocity)), ready

    @staticmethod
    def _heading_from_quat(quat: jp.ndarray) -> jp.ndarray:
        """Yaw of the IMU's -Y forward axis projected onto the world floor."""
        w, x, y, z = quat
        return jp.arctan2(2.0 * (w * z - x * y),
                         1.0 - 2.0 * (x * x + z * z))

    @staticmethod
    def _sample_uniform(
        rng: jp.ndarray,
        value_range,
        shape: tuple[int, ...],
    ) -> jp.ndarray:
        return jax.random.uniform(
            rng,
            shape,
            minval=value_range[0],
            maxval=value_range[1],
        )

    @staticmethod
    def _apply_servo_backlash(
        previous_output_target: jp.ndarray,
        motor_target: jp.ndarray,
        backlash: jp.ndarray,
    ) -> jp.ndarray:
        """Applies a stateful play operator with the given half-width."""
        return jp.clip(
            previous_output_target,
            motor_target - backlash,
            motor_target + backlash,
        )

    @staticmethod
    def _roll_pitch_quat(roll_pitch: jp.ndarray) -> jp.ndarray:
        half_roll = 0.5 * roll_pitch[0]
        half_pitch = 0.5 * roll_pitch[1]
        roll_quat = jp.array([
            jp.cos(half_roll),
            jp.sin(half_roll),
            0.0,
            0.0,
        ])
        pitch_quat = jp.array([
            jp.cos(half_pitch),
            0.0,
            jp.sin(half_pitch),
            0.0,
        ])
        return math.quat_mul(roll_quat, pitch_quat)

    def _update_attitude_estimate(
        self,
        quat: jp.ndarray,
        gyro: jp.ndarray,
        accelerometer: jp.ndarray,
    ) -> jp.ndarray:
        """Runs one firmware-equivalent 6-DoF Madgwick filter update."""
        q0, q1, q2, q3 = quat
        gx, gy, gz = gyro

        q_dot = 0.5 * jp.array([
            -q1 * gx - q2 * gy - q3 * gz,
            q0 * gx + q2 * gz - q3 * gy,
            q0 * gy - q1 * gz + q3 * gx,
            q0 * gz + q1 * gy - q2 * gx,
        ])

        accelerometer_norm_squared = jp.sum(jp.square(accelerometer))
        normalized_accelerometer = accelerometer / jp.sqrt(
            jp.maximum(accelerometer_norm_squared, 1e-12)
        )
        ax, ay, az = normalized_accelerometer

        two_q0 = 2.0 * q0
        two_q1 = 2.0 * q1
        two_q2 = 2.0 * q2
        two_q3 = 2.0 * q3
        four_q0 = 4.0 * q0
        four_q1 = 4.0 * q1
        four_q2 = 4.0 * q2
        eight_q1 = 8.0 * q1
        eight_q2 = 8.0 * q2
        q0_squared = q0 * q0
        q1_squared = q1 * q1
        q2_squared = q2 * q2
        q3_squared = q3 * q3

        correction = jp.array([
            four_q0 * q2_squared + two_q2 * ax
            + four_q0 * q1_squared - two_q1 * ay,
            four_q1 * q3_squared - two_q3 * ax
            + 4.0 * q0_squared * q1 - two_q0 * ay - four_q1
            + eight_q1 * q1_squared + eight_q1 * q2_squared
            + four_q1 * az,
            4.0 * q0_squared * q2 + two_q0 * ax
            + four_q2 * q3_squared - two_q3 * ay - four_q2
            + eight_q2 * q1_squared + eight_q2 * q2_squared
            + four_q2 * az,
            4.0 * q1_squared * q3 - two_q1 * ax
            + 4.0 * q2_squared * q3 - two_q2 * ay,
        ])
        correction_norm_squared = jp.sum(jp.square(correction))
        use_correction = (
            (accelerometer_norm_squared > 1e-12)
            & (correction_norm_squared > 1e-12)
        )
        correction_scale = jp.where(
            use_correction,
            jp.reciprocal(jp.sqrt(jp.maximum(
                correction_norm_squared, 1e-12
            ))),
            0.0,
        )
        q_dot = q_dot - self._madgwick_beta * correction * correction_scale

        quat = quat + q_dot * self.dt
        return quat / jp.maximum(jp.linalg.norm(quat), 1e-6)

    def _gravity_from_quat(self, quat: jp.ndarray) -> jp.ndarray:
        return math.rotate(
            jp.array([0.0, 0.0, -1.0]),
            self._quat_inverse(quat),
        )

    @staticmethod
    def _quat_inverse(quat: jp.ndarray) -> jp.ndarray:
        return jp.array([quat[0], -quat[1], -quat[2], -quat[3]])

    @staticmethod
    def _prewarmed_attitude_estimate(imu_quat: jp.ndarray) -> jp.ndarray:
        """Returns the settled attitude estimate available before START."""
        return imu_quat / jp.maximum(jp.linalg.norm(imu_quat), 1e-6)

    def _get_imu_quat(self, data: mjx.Data) -> jp.ndarray:
        """Returns the IMU site's local-to-world orientation quaternion."""
        quat = math.quat_mul(
            data.xquat[self._imu_body_id],
            self._imu_site_quat,
        )
        return quat / jp.maximum(jp.linalg.norm(quat), 1e-6)

    def _get_imu_linear_velocity_world(
        self,
        data_before: mjx.Data,
        data_after: mjx.Data,
    ) -> jp.ndarray:
        """Returns finite-difference velocity at the IMU site's position."""
        return (
            data_after.site_xpos[self._imu_site_id]
            - data_before.site_xpos[self._imu_site_id]
        ) / self.dt

    def _get_imu_signals(
        self,
        data: mjx.Data,
    ) -> tuple[jp.ndarray, jp.ndarray]:
        """Returns ideal projected gravity and gyro in the IMU site frame."""
        root_quat = data.qpos[3:7]
        root_quat = root_quat / jp.linalg.norm(root_quat)
        imu_xmat = data.site_xmat[self._imu_site_id]
        gravity_local = imu_xmat.T @ jp.array([0.0, 0.0, -1.0])
        gyro_world = math.rotate(
            data.qvel[3:6],
            root_quat,
        )
        gyro_local = imu_xmat.T @ gyro_world
        return gravity_local, gyro_local

    @staticmethod
    def _wrap_phase(phase: jp.ndarray) -> jp.ndarray:
        return jp.mod(phase + jp.pi, 2.0 * jp.pi) - jp.pi

    def _desired_foot_heights(
        self,
        phase: jp.ndarray,
    ) -> jp.ndarray:
        foot_phases = phase + jp.array([0.0, jp.pi])
        swing = jp.maximum(jp.sin(foot_phases), 0.0)
        return self._natural_clearance_scale * jp.square(swing)

    def _get_swing_foot_forward_reward(
        self,
        foot_positions_world: jp.ndarray,
        body_position_world: jp.ndarray,
        body_forward_xy: jp.ndarray,
        phase: jp.ndarray,
        foot_velocities_world: jp.ndarray,
        body_velocity_world: jp.ndarray,
        motion_command: jp.ndarray,
        body_yaw_rate: jp.ndarray,
        contacts: jp.ndarray,
        toe_clearance: jp.ndarray,
        heel_clearance: jp.ndarray,
    ) -> tuple[jp.ndarray, jp.ndarray]:
        foot_offsets_xy = (
            foot_positions_world[:, :2] - body_position_world[:2]
        )
        foot_forward_positions = foot_offsets_xy @ body_forward_xy
        foot_phases = phase + jp.array([0.0, jp.pi])
        phase_sin = jp.sin(foot_phases)
        swing = jp.square(jp.maximum(phase_sin, 0.0))
        yaw_tangent = jp.stack(
            [-foot_offsets_xy[:, 1], foot_offsets_xy[:, 0]], axis=1
        )
        # Differentiate in the rotating body frame: a planted stance foot
        # retracts relative to both body translation and body rotation.
        foot_relative_velocity = (
            foot_velocities_world[:, :2] - body_velocity_world[:2]
            - body_yaw_rate * yaw_tangent
        )
        # Rigid-body command at each foot includes yaw. This gives opposite
        # fore/aft stepping directions for the two feet during a pivot turn.
        yaw_velocity = motion_command[1] * yaw_tangent
        desired_velocity = motion_command[0] * body_forward_xy + yaw_velocity
        speed = jp.linalg.norm(desired_velocity, axis=1)
        direction = desired_velocity / jp.maximum(speed[:, None], 1e-3)
        relative_speed = jp.sum(foot_relative_velocity * direction, axis=1)
        # Foot progress earns credit only for a lifted swing foot supported by
        # the opposite foot. Dense unloading/lift rewards remain independent,
        # so discovering liftoff does not depend on this contact gate.
        clearance = jp.minimum(toe_clearance, heel_clearance)
        height_fraction = jp.clip(clearance / self._swing_progress_clearance, 0.0, 1.0)
        clearance_credit = jp.square(height_fraction) * (3.0 - 2.0 * height_fraction)
        supported_swing = swing * (~contacts) * contacts[::-1] * clearance_credit
        progress = jp.tanh(relative_speed / jp.maximum(2.0 * speed, 1e-3))
        reward = self._swing_foot_forward_reward_weight * jp.sum(supported_swing * progress)
        return reward, foot_forward_positions
