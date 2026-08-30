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
        'target_velocity': 0.06,
        'action_scale': 0.50,
        'step_frequency': 0.8,

        # Four commands = 80 ms of controller-known history at 50 Hz.  This
        # is the policy's only proxy for joint state on feedback-free servos.
        'command_history_length': 4,

        # Hidden actuator uncertainty.  The policy sees the command it sent,
        # not the perturbed/lagged command used by the simulation.
        'servo_response': 0.30,
        'action_noise_scale': 0.02,

        # Reset randomization.
        'reset_noise_scale': 0.002,

        # IMU observation model.  Projected gravity is a unit-vector estimate
        # from accelerometer/gyro fusion; gyro values are in rad/s.
        'gravity_noise_scale': 0.02,
        'gravity_bias_scale': 0.01,
        'gyro_noise_scale': 0.03,
        'gyro_bias_scale': 0.02,
        'gyro_obs_scale': 0.25,

        # Task rewards.
        'velocity_tracking_weight': 2.0,
        'tracking_sigma': 0.0025,
        'sideways_velocity_cost_weight': 0.05,
        'yaw_rate_cost_weight': 0.20,
        'heading_cost_weight': 0.50,
        'orientation_cost_weight': 0.30,
        'action_rate_cost_weight': 0.02,
        # The Pico maps a pi-radian servo range to 2000 us.  Penalize command
        # changes inside the measured MG90S deadband so the policy learns to
        # either hold position or make a change the real servo can execute.
        'servo_deadband_us': 10.0,
        'servo_deadband_cost_weight': 0.05,
        'healthy_reward': 0.20,
        'vertical_velocity_cost_weight': 0.05,

        # Gait shaping copied from the successful simple task.
        'foot_phase_reward_weight': 1.0,
        'swing_foot_height': 0.012,
        'foot_height_tracking_sigma': 2.5e-5,
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

        self._target_velocity = config.target_velocity
        self._action_scale = config.action_scale
        self._step_frequency = config.step_frequency
        self._servo_response = config.servo_response
        self._action_noise_scale = config.action_noise_scale
        self._reset_noise_scale = config.reset_noise_scale

        self._gravity_noise_scale = config.gravity_noise_scale
        self._gravity_bias_scale = config.gravity_bias_scale
        self._gyro_noise_scale = config.gyro_noise_scale
        self._gyro_bias_scale = config.gyro_bias_scale
        self._gyro_obs_scale = config.gyro_obs_scale

        self._velocity_tracking_weight = config.velocity_tracking_weight
        self._tracking_sigma = config.tracking_sigma
        self._sideways_velocity_cost_weight = (
            config.sideways_velocity_cost_weight
        )
        self._yaw_rate_cost_weight = config.yaw_rate_cost_weight
        self._heading_cost_weight = config.heading_cost_weight
        self._orientation_cost_weight = config.orientation_cost_weight
        self._action_rate_cost_weight = config.action_rate_cost_weight
        self._servo_deadband_us = config.servo_deadband_us
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

        self._default_pose = self._mjx_model.qpos0[7:]
        default_data = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, default_data)
        self._nominal_base_height = default_data.subtree_com[
            self._body_idx, 2
        ]
        self._ctrl_min = self._mjx_model.actuator_ctrlrange[:, 0]
        self._ctrl_max = self._mjx_model.actuator_ctrlrange[:, 1]

        # Robot convention: +X sideways, -Y forward, +Z up.
        self._forward_world = jp.array([0.0, -1.0, 0.0])
        self._forward_world_xy = jp.array([0.0, -1.0])

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
            qpos_key,
            qvel_key,
            phase_key,
            gravity_bias_key,
            gyro_bias_key,
            obs_key,
            step_key,
        ) = jax.random.split(rng, 7)

        low, high = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self._mjx_model.qpos0 + jax.random.uniform(
            qpos_key,
            (self._mjx_model.nq,),
            minval=low,
            maxval=high,
        )
        root_quat = qpos[3:7]
        root_quat = root_quat / jp.linalg.norm(root_quat)
        qpos = qpos.at[3:7].set(root_quat)

        qvel = jax.random.uniform(
            qvel_key,
            (self._mjx_model.nv,),
            minval=low,
            maxval=high,
        )

        data = mujoco_playground._src.mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
        )
        data = mjx.forward(self._mjx_model, data)

        phase = jax.random.uniform(
            phase_key,
            minval=-jp.pi,
            maxval=jp.pi,
        )
        command_history = jp.zeros((self._history_len, self._action_dim))
        servo_action = jp.zeros((self._action_dim,))

        gravity_bias = jax.random.uniform(
            gravity_bias_key,
            (3,),
            minval=-self._gravity_bias_scale,
            maxval=self._gravity_bias_scale,
        )
        gyro_bias = jax.random.uniform(
            gyro_bias_key,
            (3,),
            minval=-self._gyro_bias_scale,
            maxval=self._gyro_bias_scale,
        )

        gravity_local, gyro_local = self._get_imu_signals(data)
        observation = self._get_observation(
            command_history=command_history,
            gravity_local=gravity_local,
            gyro_local=gyro_local,
            phase=phase,
            gravity_bias=gravity_bias,
            gyro_bias=gyro_bias,
            noise_key=obs_key,
        )

        zero = jp.array(0.0)
        metrics = {
            'reward_velocity_tracking': zero,
            'reward_foot_phase': zero,
            'reward_alive': zero,
            'cost_sideways_velocity': zero,
            'cost_yaw_rate': zero,
            'cost_heading': zero,
            'cost_orientation': zero,
            'cost_action_rate': zero,
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
        }

        return mjx_env.State(
            data=data,
            obs=observation,
            reward=zero,
            done=zero,
            metrics=metrics,
            info={
                'command_history': command_history,
                'servo_action': servo_action,
                'phase': phase,
                'gravity_bias': gravity_bias,
                'gyro_bias': gyro_bias,
                'rng': step_key,
            },
        )

    def step(
        self,
        state: mjx_env.State,
        action: jp.ndarray,
    ) -> mjx_env.State:
        action_noise_key, obs_key, step_key = jax.random.split(
            state.info['rng'], 3
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
            action_delta_us / self._servo_deadband_us,
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

        actuation_error = (
            jax.random.normal(action_noise_key, command.shape)
            * self._action_noise_scale
        )
        uncertain_action = jp.clip(
            command + actuation_error,
            -1.0,
            1.0,
        )
        servo_action = (
            self._servo_response * uncertain_action
            + (1.0 - self._servo_response) * state.info['servo_action']
        )

        motor_targets = self._default_pose + self._action_scale * servo_action
        motor_targets = jp.clip(
            motor_targets,
            self._ctrl_min,
            self._ctrl_max,
        )

        new_command_history = jp.roll(command_history, shift=-1, axis=0)
        new_command_history = new_command_history.at[-1].set(command)

        data0 = state.data
        data = mjx_env.step(
            self._mjx_model,
            data0,
            motor_targets,
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

        foot_velocities_world = (
            foot_positions_after - foot_positions_before
        ) / self.dt
        stance_weights = (
            desired_foot_heights <= 1e-6
        ).astype(foot_heights.dtype)
        foot_slip_cost = (
            self._foot_slip_cost_weight
            * jp.sum(
                stance_weights
                * jp.sum(
                    jp.square(foot_velocities_world[:, :2]),
                    axis=1,
                )
            )
        )

        # These global quantities shape training but never enter observation.
        root_angular_velocity_local = data.qvel[3:6]
        root_angular_velocity_world = math.rotate(
            root_angular_velocity_local,
            root_quat,
        )
        yaw_rate = root_angular_velocity_world[2]
        yaw_rate_cost = self._yaw_rate_cost_weight * jp.square(yaw_rate)

        body_forward_world = math.rotate(self._forward_world, root_quat)
        body_forward_xy = body_forward_world[:2]
        body_forward_xy = body_forward_xy / jp.maximum(
            jp.linalg.norm(body_forward_xy),
            1e-6,
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
        base_height_cost = (
            self._base_height_cost_weight
            * jp.square(base_height - self._nominal_base_height)
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
            + healthy_reward
            - sideways_velocity_cost
            - yaw_rate_cost
            - heading_cost
            - tilt_cost
            - action_rate_cost
            - servo_deadband_cost
            - vertical_velocity_cost
            - foot_slip_cost
            - base_height_cost
            - energy_cost
        )

        # Only these two state-derived signals are exposed, and both are
        # obtainable from the LSM6DSO.  All reward state above is privileged.
        _, gyro_local = self._get_imu_signals(data)
        observation = self._get_observation(
            command_history=new_command_history,
            gravity_local=gravity_local,
            gyro_local=gyro_local,
            phase=phase,
            gravity_bias=state.info['gravity_bias'],
            gyro_bias=state.info['gyro_bias'],
            noise_key=obs_key,
        )

        state.metrics.update(
            reward_velocity_tracking=velocity_tracking_reward,
            reward_foot_phase=foot_phase_reward,
            reward_alive=healthy_reward,
            cost_sideways_velocity=-sideways_velocity_cost,
            cost_yaw_rate=-yaw_rate_cost,
            cost_heading=-heading_cost,
            cost_orientation=-tilt_cost,
            cost_action_rate=-action_rate_cost,
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
        )

        return state.replace(
            data=data,
            obs=observation,
            reward=reward,
            done=done,
            info={
                **state.info,
                'command_history': new_command_history,
                'servo_action': servo_action,
                'phase': phase,
                'rng': step_key,
            },
        )

    def _get_observation(
        self,
        command_history: jp.ndarray,
        gravity_local: jp.ndarray,
        gyro_local: jp.ndarray,
        phase: jp.ndarray,
        gravity_bias: jp.ndarray,
        gyro_bias: jp.ndarray,
        noise_key: jp.ndarray,
    ) -> jp.ndarray:
        """Builds the 41-value observation available on the real robot.

        Layout:
          0:32  normalized command history, oldest to newest
          32:35 unit gravity estimate in IMU/body coordinates
          35:38 scaled gyroscope in IMU/body coordinates
          38:40 sin/cos gait clock
          40    commanded speed in m/s
        """
        gravity_key, gyro_key = jax.random.split(noise_key)

        gravity_estimate = (
            gravity_local
            + gravity_bias
            + jax.random.normal(gravity_key, (3,))
            * self._gravity_noise_scale
        )
        gravity_estimate = gravity_estimate / jp.maximum(
            jp.linalg.norm(gravity_estimate),
            1e-6,
        )

        gyro_measurement = (
            gyro_local
            + gyro_bias
            + jax.random.normal(gyro_key, (3,)) * self._gyro_noise_scale
        )

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
    def _quat_inverse(quat: jp.ndarray) -> jp.ndarray:
        return jp.array([quat[0], -quat[1], -quat[2], -quat[3]])

    def _get_imu_signals(
        self,
        data: mjx.Data,
    ) -> tuple[jp.ndarray, jp.ndarray]:
        """Returns ideal projected gravity and gyro in the robot body frame."""
        root_quat = data.qpos[3:7]
        root_quat = root_quat / jp.linalg.norm(root_quat)
        gravity_local = math.rotate(
            jp.array([0.0, 0.0, -1.0]),
            self._quat_inverse(root_quat),
        )
        gyro_local = data.qvel[3:6]
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
