"""Rick walking task with observations suitable for the real robot.

The policy observation contains only values available at deployment:

* recent normalized servo commands (known by the controller),
* a body-frame gravity estimate and angular velocity from the LSM6DSO,
* the controller's gait clock, and
* the commanded forward speed.

Joint state, root velocity, heading, foot state, and contact state are used only
for rewards, termination, and diagnostics during training.
"""

import jax
from jax import numpy as jp
from ml_collections import config_dict

import mujoco
from mujoco import mjx
from brax import math

import mujoco_playground
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.rick import rick_constants as consts


_SERVO_US_PER_RADIAN = 2000.0 / 3.141592653589793


def default_config() -> config_dict.ConfigDict:
    return config_dict.ConfigDict({
        # Simulation / control.
        'ctrl_dt': 0.02,
        'sim_dt': 0.002,
        'episode_length': 1000,
        'action_repeat': 1,
        'impl': 'brax',

        # Policy command and gait parameters.
        'target_velocity': 0.068,
        'action_scale': 0.40,
        'step_frequency': 0.8,

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
        'center_of_mass_offset_scale': (0.005, 0.005, 0.003),  # m.

        # Reset randomization.  Keep units separate: a single scale applied to
        # the free joint, quaternion, and leg joints is not physically useful.
        'root_position_noise_scale': (0.002, 0.002, 0.001),  # m.
        'root_tilt_noise_scale': 0.08,  # rad
        'joint_position_noise_scale': 0.03,  # rad.
        'root_linear_velocity_noise_scale': 0.02,  # m/s.
        'root_angular_velocity_noise_scale': 0.10,  # rad/s.
        'joint_velocity_noise_scale': 0.10,  # rad/s.

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

        # Task rewards.
        'velocity_tracking_weight': 2.0,
        'tracking_sigma': 0.0025,
        'sideways_velocity_cost_weight': 0.05,
        'yaw_rate_cost_weight': 0.20,
        'heading_cost_weight': 0.50,
        'orientation_cost_weight': 0.75,
        'roll_pitch_rate_cost_weight': 0.05,
        'action_rate_cost_weight': 0.05,
        'action_acceleration_cost_weight': 0.02,
        # Penalize changes inside the episode's sampled MG90S deadband so the
        # policy learns to hold or make a change the real servo can execute.
        'servo_deadband_cost_weight': 0.01,
        'healthy_reward': 0.20,
        'vertical_velocity_cost_weight': 0.05,

        # Gait shaping copied from the successful simple task.
        'foot_phase_reward_weight': 1.0,
        'swing_foot_height': 0.015,
        'foot_height_tracking_sigma': 2.5e-5,
        # During each swing, move the foot 30 mm from behind to ahead of its
        # nominal body-relative position.  At the target speed/frequency this
        # roughly matches the distance the body travels during one stance.
        'swing_foot_forward_reward_weight': 0,
        'swing_foot_forward_distance': 0.030,
        'swing_foot_forward_tracking_sigma': 1.0e-4,  # (10 mm)^2.
        'foot_slip_cost_weight': 20.0,
        'base_height_cost_weight': 0.10,
        'base_height_tracking_sigma': 4.0e-4,
        'energy_cost_weight': 0.005,

        # Termination.
        'terminate_when_unhealthy': True,
        'healthy_z_range': (0.05, 0.20),
        'minimum_upright': 0.50,
    })


class Joystick(mjx_env.MjxEnv):
    """Tracks a forward command using deployment-compatible observations."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: dict = None,
    ):
        super().__init__(config, config_overrides)

        path = consts.task_to_xml(task)
        self._mj_model = mujoco.MjModel.from_xml_path(path.as_posix())
        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self._mj_model.opt.iterations = 10
        self._mj_model.opt.ls_iterations = 6
        self._mjx_model = mjx.put_model(self._mj_model)

        self._action_dim = self._mjx_model.nu
        self._history_len = config.command_history_length
        if self._history_len < 2:
            raise ValueError("command_history_length must be at least 2")

        self._target_velocity = config.target_velocity
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
        self._sideways_velocity_cost_weight = (
            config.sideways_velocity_cost_weight
        )
        self._yaw_rate_cost_weight = config.yaw_rate_cost_weight
        self._heading_cost_weight = config.heading_cost_weight
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
        self._swing_foot_height = config.swing_foot_height
        self._foot_height_tracking_sigma = (
            config.foot_height_tracking_sigma
        )
        self._swing_foot_forward_reward_weight = (
            config.swing_foot_forward_reward_weight
        )
        self._swing_foot_forward_distance = (
            config.swing_foot_forward_distance
        )
        self._swing_foot_forward_tracking_sigma = (
            config.swing_foot_forward_tracking_sigma
        )
        self._foot_slip_cost_weight = config.foot_slip_cost_weight
        self._base_height_cost_weight = config.base_height_cost_weight
        self._base_height_tracking_sigma = (
            config.base_height_tracking_sigma
        )
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

        self._default_pose = self._mjx_model.qpos0[7:]
        default_data = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, default_data)
        self._nominal_base_height = default_data.subtree_com[
            self._body_idx, 2
        ]
        default_foot_offsets = (
            jp.array(default_data.site_xpos)[self._feet_site_ids]
            - jp.array(default_data.xpos[self._body_idx])
        )
        self._nominal_foot_forward_positions = (
            default_foot_offsets @ self._forward_world
        )
        self._ctrl_min = self._mjx_model.actuator_ctrlrange[:, 0]
        self._ctrl_max = self._mjx_model.actuator_ctrlrange[:, 1]
        actuator_joint_ids = self._mj_model.actuator_trnid[:, 0]
        self._actuated_dof_ids = jp.array(
            self._mj_model.jnt_dofadr[actuator_joint_ids]
        )

    @property
    def xml_path(self) -> str:
        return consts.task_to_xml("flat_terrain").as_posix()

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

        qpos = self._mjx_model.qpos0
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

        # The Pico runs Madgwick continuously while stopped and preserves the
        # settled estimate on START.  Begin from the simulated IMU site's
        # current attitude to match that prewarmed deployment behavior.
        imu_quat = self._prewarmed_attitude_estimate(
            self._get_imu_quat(data)
        )
        gravity_estimate = self._gravity_from_quat(imu_quat)
        _, gyro_local = self._get_imu_signals(data)
        gyro_measurement = (
            gyro_local
            + gyro_bias
            + jax.random.normal(initial_gyro_key, (3,))
            * self._gyro_noise_scale
        )
        observation = self._get_observation(
            command_history=command_history,
            gravity_estimate=gravity_estimate,
            gyro_measurement=gyro_measurement,
            phase=phase,
        )

        zero = jp.array(0.0)
        metrics = {
            'reward_velocity_tracking': zero,
            'reward_foot_phase': zero,
            'reward_swing_foot_forward': zero,
            'reward_alive': zero,
            'cost_sideways_velocity': zero,
            'cost_yaw_rate': zero,
            'cost_heading': zero,
            'cost_orientation': zero,
            'cost_roll_pitch_rate': zero,
            'cost_action_rate': zero,
            'cost_action_acceleration': zero,
            'cost_servo_deadband': zero,
            'cost_vertical_velocity': zero,
            'cost_foot_slip': zero,
            'cost_base_height': zero,
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
            'desired_left_foot_height': zero,
            'desired_right_foot_height': zero,
            'left_foot_forward_position': zero,
            'right_foot_forward_position': zero,
            'desired_left_foot_forward_position': zero,
            'desired_right_foot_forward_position': zero,
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
                'previous_imu_velocity_world': jp.zeros((3,)),
                'accelerometer_bias': accelerometer_bias,
                'gyro_bias': gyro_bias,
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

        data0 = state.data
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

        com_before = data0.subtree_com[self._body_idx]
        com_after = data.subtree_com[self._body_idx]
        velocity_world = (com_after - com_before) / self.dt
        forward_velocity = -velocity_world[1]
        sideways_velocity = velocity_world[0]

        velocity_tracking_reward = (
            self._velocity_tracking_weight
            * jp.exp(
                -jp.square(forward_velocity - self._target_velocity)
                / self._tracking_sigma
            )
        )
        sideways_velocity_cost = (
            self._sideways_velocity_cost_weight
            * jp.square(sideways_velocity)
            / self._tracking_sigma
        )

        desired_foot_heights = self._desired_foot_heights(phase)
        foot_positions_before = data0.site_xpos[self._feet_site_ids]
        foot_positions_after = data.site_xpos[self._feet_site_ids]
        foot_heights = jp.maximum(foot_positions_after[:, 2], 0.0)
        foot_height_error = jp.mean(
            jp.square(foot_heights - desired_foot_heights)
        )
        foot_phase_reward = (
            self._foot_phase_reward_weight
            * jp.exp(
                -foot_height_error
                / self._foot_height_tracking_sigma
            )
        )

        # Track a fore-aft swing trajectory relative to the body.  Whole-body
        # translation cannot satisfy this objective: the airborne foot must
        # move from behind the torso to ahead of it before touchdown.
        (
            swing_foot_forward_reward,
            foot_forward_positions,
            desired_foot_forward_positions,
        ) = self._get_swing_foot_forward_reward(
            foot_positions_after,
            data.xpos[self._body_idx],
            body_forward_xy,
            phase,
        )

        foot_velocities_world = (
            foot_positions_after - foot_positions_before
        ) / self.dt
        foot_contacts = (
            data.sensordata[self._foot_contact_sensor_adrs] > 0
        )
        foot_slip_cost = self._get_foot_slip_cost(
            foot_velocities_world,
            desired_foot_heights,
            foot_contacts,
        )

        # These global quantities shape training but never enter observation.
        root_angular_velocity_local = data.qvel[3:6]
        root_angular_velocity_world = math.rotate(
            root_angular_velocity_local,
            root_quat,
        )
        yaw_rate = root_angular_velocity_world[2]
        yaw_rate_cost = self._yaw_rate_cost_weight * jp.square(yaw_rate)
        roll_pitch_rate_cost = (
            self._roll_pitch_rate_cost_weight
            * jp.sum(jp.square(root_angular_velocity_local[:2]))
        )

        heading_alignment = jp.dot(
            body_forward_xy,
            self._forward_world_xy,
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
        target_base_height = (
            self._nominal_base_height
            + state.info['center_of_mass_offset'][2]
        )
        base_height_cost = (
            self._base_height_cost_weight
            * jp.square(base_height - target_base_height)
            / self._base_height_tracking_sigma
        )
        energy_cost = (
            self._energy_cost_weight
            * jp.sum(jp.abs(data.actuator_force * data.qvel[6:]))
        )

        min_z, max_z = self._healthy_z_range
        healthy_height = (base_height >= min_z) & (base_height <= max_z)
        healthy_orientation = uprightness >= self._minimum_upright
        is_healthy = healthy_height & healthy_orientation
        is_healthy_float = is_healthy.astype(jp.float32)
        healthy_reward = self._healthy_reward * is_healthy_float
        done = (
            1.0 - is_healthy_float
            if self._terminate_when_unhealthy
            else jp.array(0.0)
        )

        reward = (
            velocity_tracking_reward
            + foot_phase_reward
            + swing_foot_forward_reward
            + healthy_reward
            - sideways_velocity_cost
            - yaw_rate_cost
            - heading_cost
            - tilt_cost
            - roll_pitch_rate_cost
            - action_rate_cost
            - action_acceleration_cost
            - servo_deadband_cost
            - vertical_velocity_cost
            - foot_slip_cost
            - base_height_cost
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
            + jax.random.normal(gyro_noise_key, (3,))
            * self._gyro_noise_scale
        )
        imu_quat = self._update_attitude_estimate(
            state.info['imu_quat'],
            gyro_measurement,
            accelerometer_measurement,
        )
        gravity_estimate = self._gravity_from_quat(imu_quat)

        # Only the filtered gravity estimate and raw gyro are exposed.  All
        # reward state above remains privileged to training.
        observation = self._get_observation(
            command_history=new_command_history,
            gravity_estimate=gravity_estimate,
            gyro_measurement=gyro_measurement,
            phase=phase,
        )

        state.metrics.update(
            reward_velocity_tracking=velocity_tracking_reward,
            reward_foot_phase=foot_phase_reward,
            reward_swing_foot_forward=swing_foot_forward_reward,
            reward_alive=healthy_reward,
            cost_sideways_velocity=-sideways_velocity_cost,
            cost_yaw_rate=-yaw_rate_cost,
            cost_heading=-heading_cost,
            cost_orientation=-tilt_cost,
            cost_roll_pitch_rate=-roll_pitch_rate_cost,
            cost_action_rate=-action_rate_cost,
            cost_action_acceleration=-action_acceleration_cost,
            cost_servo_deadband=-servo_deadband_cost,
            cost_vertical_velocity=-vertical_velocity_cost,
            cost_foot_slip=-foot_slip_cost,
            cost_base_height=-base_height_cost,
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
            desired_left_foot_height=desired_foot_heights[0],
            desired_right_foot_height=desired_foot_heights[1],
            left_foot_forward_position=foot_forward_positions[0],
            right_foot_forward_position=foot_forward_positions[1],
            desired_left_foot_forward_position=(
                desired_foot_forward_positions[0]
            ),
            desired_right_foot_forward_position=(
                desired_foot_forward_positions[1]
            ),
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
                'previous_imu_velocity_world': imu_velocity_world,
                'rng': step_key,
            },
        )

    def _get_foot_slip_cost(
        self,
        foot_velocities_world: jp.ndarray,
        desired_foot_heights: jp.ndarray,
        foot_contacts: jp.ndarray,
    ) -> jp.ndarray:
        """Penalizes tangential velocity only for contacting stance feet."""
        stance_contacts = (
            (desired_foot_heights <= 1e-6) & foot_contacts
        ).astype(foot_velocities_world.dtype)
        tangential_speed_squared = jp.sum(
            jp.square(foot_velocities_world[:, :2]),
            axis=1,
        )
        return self._foot_slip_cost_weight * jp.sum(
            stance_contacts * tangential_speed_squared
        )

    def _body_ipos_with_com_offset(
        self,
        center_of_mass_offset: jp.ndarray,
    ) -> jp.ndarray:
        """Applies a shared COM offset without modifying the world body."""
        return self._mjx_model.body_ipos.at[1:].add(
            center_of_mass_offset
        )

    def _get_observation(
        self,
        command_history: jp.ndarray,
        gravity_estimate: jp.ndarray,
        gyro_measurement: jp.ndarray,
        phase: jp.ndarray,
    ) -> jp.ndarray:
        """Builds the deployment observation (105 values by default).

        With the default 12-frame history, the layout is:
          0:96   normalized command history, oldest to newest
          96:99  unit gravity estimate in IMU coordinates
          99:102 scaled gyroscope in IMU coordinates
          102:104 sin/cos gait clock
          104     commanded speed in m/s
        """
        clock = jp.array([jp.sin(phase), jp.cos(phase)])
        return jp.clip(
            jp.concatenate([
                command_history.flatten(),
                gravity_estimate,
                self._gyro_obs_scale * gyro_measurement,
                clock,
                jp.array([self._target_velocity]),
            ]),
            -10.0,
            10.0,
        )

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
        return self._swing_foot_height * jp.square(swing)

    def _desired_foot_forward_positions(
        self,
        phase: jp.ndarray,
    ) -> jp.ndarray:
        """Returns body-relative targets, from rear to front during swing."""
        foot_phases = phase + jp.array([0.0, jp.pi])
        return (
            self._nominal_foot_forward_positions
            - 0.5
            * self._swing_foot_forward_distance
            * jp.cos(foot_phases)
        )

    def _get_swing_foot_forward_reward(
        self,
        foot_positions_world: jp.ndarray,
        body_position_world: jp.ndarray,
        body_forward_xy: jp.ndarray,
        phase: jp.ndarray,
    ) -> tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
        """Rewards the swing foot for following a body-relative forward arc."""
        foot_offsets_xy = (
            foot_positions_world[:, :2] - body_position_world[:2]
        )
        foot_forward_positions = foot_offsets_xy @ body_forward_xy
        desired_foot_forward_positions = (
            self._desired_foot_forward_positions(phase)
        )

        foot_phases = phase + jp.array([0.0, jp.pi])
        swing_weights = (jp.sin(foot_phases) > 0.0).astype(
            foot_positions_world.dtype
        )
        swing_count = jp.sum(swing_weights)
        tracking_error = jp.sum(
            swing_weights
            * jp.square(
                foot_forward_positions - desired_foot_forward_positions
            )
        ) / jp.maximum(swing_count, 1.0)
        reward = (
            self._swing_foot_forward_reward_weight
            * jp.minimum(swing_count, 1.0)
            * jp.exp(
                -tracking_error
                / self._swing_foot_forward_tracking_sigma
            )
        )
        return (
            reward,
            foot_forward_positions,
            desired_foot_forward_positions,
        )
