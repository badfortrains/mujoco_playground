import jax
from jax import numpy as jp
from ml_collections import config_dict

import mujoco
from mujoco import mjx
from brax import math

import mujoco_playground
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.rick import rick_constants as consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.ConfigDict({
        # Simulation / control.
        'ctrl_dt': 0.02,
        'sim_dt': 0.002,
        'episode_length': 1000,
        'action_repeat': 1,
        'impl': 'brax',

        # -------------------------------------------------------------
        # Simplified locomotion command
        # -------------------------------------------------------------
        # Keep this FIXED initially.
        #
        # Once straight walking works, you can randomize it again.
        'target_velocity': 0.06,  # m/s in world -Y direction

        # Policy action is interpreted as an offset from the default pose.
        # +/- 1 action -> +/- 0.35 rad (~20 degrees)
        'action_scale': 0.35,

        # Clock supplied to the policy.
        'step_frequency': 0.8,

        # -------------------------------------------------------------
        # Rewards
        # -------------------------------------------------------------

        # Main task: match forward speed.  Lateral motion is penalized
        # separately so the torso can sway while transferring support.
        'velocity_tracking_weight': 2.0,

        # Width of exponential velocity tracking reward.
        #
        # exp(-error_squared / tracking_sigma)
        #
        # At target=0.06:
        # standing reward ~= exp(-0.0036 / 0.0025) ~= 0.24
        # instead of ~0.70 in your old setup.
        'tracking_sigma': 0.0025,

        # Mild lateral-velocity regularization.  This is normalized by
        # tracking_sigma below and is deliberately much softer than putting
        # lateral velocity inside the exponential tracking reward.
        'sideways_velocity_cost_weight': 0.05,

        # Explicitly discourage turning.
        'yaw_rate_cost_weight': 0.20,

        # Explicitly keep body facing original -Y direction.
        'heading_cost_weight': 0.50,

        # Stay upright.
        'orientation_cost_weight': 0.30,

        # Keep motion smooth, but MUCH weaker than before.
        'action_rate_cost_weight': 0.02,

        # Small survival incentive.
        # We do not want standing still to dominate the task reward.
        'healthy_reward': 0.20,

        # Optional small vertical-motion penalty.
        'vertical_velocity_cost_weight': 0.05,

        # -------------------------------------------------------------
        # Gait shaping
        # -------------------------------------------------------------

        # Track alternating left/right swing-foot height targets.  The
        # targets have a 50% duty cycle and are 180 degrees out of phase.
        'foot_phase_reward_weight': 1.0,
        'swing_foot_height': 0.012,  # 12 mm for this 145 mm-tall robot.
        'foot_height_tracking_sigma': 2.5e-5,  # (5 mm)^2.

        # Penalize planar motion of the foot whose phase says it should be
        # supporting the robot.  This directly attacks the shuffling local
        # optimum.  Units are inverse (m/s)^2.
        'foot_slip_cost_weight': 20.0,

        # Discourage a permanently crouched solution while leaving room for
        # normal vertical COM motion during a step.
        'base_height_cost_weight': 0.10,
        'base_height_tracking_sigma': 4.0e-4,  # (20 mm)^2.

        # Light mechanical-power penalty to suppress high-frequency motion.
        'energy_cost_weight': 0.005,

        # -------------------------------------------------------------
        # Termination
        # -------------------------------------------------------------
        'terminate_when_unhealthy': True,
        'healthy_z_range': (0.05, 0.20),

        # -gravity_local[z] == 1 when upright.
        # 0.5 corresponds roughly to 60 degrees from upright.
        'minimum_upright': 0.50,

        # Observation scaling.
        'joint_velocity_obs_scale': 0.10,
        'gyro_obs_scale': 0.25,
    })


class Joystick(mjx_env.MjxEnv):

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

        # -------------------------------------------------------------
        # Configuration
        # -------------------------------------------------------------

        self._target_velocity = config.target_velocity
        self._action_scale = config.action_scale
        self._step_frequency = config.step_frequency

        self._velocity_tracking_weight = (
            config.velocity_tracking_weight
        )
        self._tracking_sigma = config.tracking_sigma

        self._sideways_velocity_cost_weight = (
            config.sideways_velocity_cost_weight
        )

        self._yaw_rate_cost_weight = (
            config.yaw_rate_cost_weight
        )
        self._heading_cost_weight = (
            config.heading_cost_weight
        )
        self._orientation_cost_weight = (
            config.orientation_cost_weight
        )
        self._action_rate_cost_weight = (
            config.action_rate_cost_weight
        )
        self._vertical_velocity_cost_weight = (
            config.vertical_velocity_cost_weight
        )

        self._foot_phase_reward_weight = (
            config.foot_phase_reward_weight
        )
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

        self._healthy_reward = config.healthy_reward
        self._terminate_when_unhealthy = (
            config.terminate_when_unhealthy
        )
        self._healthy_z_range = config.healthy_z_range
        self._minimum_upright = config.minimum_upright

        self._joint_velocity_obs_scale = (
            config.joint_velocity_obs_scale
        )
        self._gyro_obs_scale = config.gyro_obs_scale

        # -------------------------------------------------------------
        # Robot indices
        # -------------------------------------------------------------

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

        # qpos layout with a free joint:
        #
        #   0:3   root position
        #   3:7   root quaternion
        #   7:    actuated joint positions
        #
        # Start all policies around the XML's nominal joint pose.
        self._default_pose = self._mjx_model.qpos0[7:]
        self._nominal_base_height = self._mjx_model.qpos0[2]

        # Position-actuator control limits.
        self._ctrl_min = self._mjx_model.actuator_ctrlrange[:, 0]
        self._ctrl_max = self._mjx_model.actuator_ctrlrange[:, 1]

        # Coordinate convention from your robot:
        #
        # +X = sideways
        # -Y = forward
        # +Z = up
        self._forward_world = jp.array([0.0, -1.0, 0.0])
        self._forward_world_xy = jp.array([0.0, -1.0])

    # ------------------------------------------------------------------
    # Standard environment properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, rng: jp.ndarray) -> mjx_env.State:
        _, phase_key, step_key = jax.random.split(rng, 3)

        # A random starting phase prevents the policy from depending on a
        # reset-only transient.  The two feet are offset by pi below.
        phase = jax.random.uniform(
            phase_key,
            minval=-jp.pi,
            maxval=jp.pi,
        )

        qpos = self._mjx_model.qpos0

        # Normalize free-joint quaternion.
        root_quat = qpos[3:7]
        root_quat = root_quat / jp.linalg.norm(root_quat)
        qpos = qpos.at[3:7].set(root_quat)

        qvel = jp.zeros((self._mjx_model.nv,))

        data = mujoco_playground._src.mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
        )

        data = mjx.forward(self._mjx_model, data)

        last_action = jp.zeros((self._action_dim,))

        obs = self._get_obs(
            data=data,
            last_action=last_action,
            phase=phase,
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
            obs=obs,
            reward=zero,
            done=zero,
            metrics=metrics,
            info={
                'last_action': last_action,
                'phase': phase,
                'rng': step_key,
            },
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        state: mjx_env.State,
        action: jp.ndarray,
    ) -> mjx_env.State:

        _, step_key = jax.random.split(state.info['rng'])

        # -------------------------------------------------------------
        # 1. Action processing
        # -------------------------------------------------------------

        action = jp.clip(action, -1.0, 1.0)

        last_action = state.info['last_action']

        # Do NOT smooth the action here.
        #
        # Policy chooses small offsets around a nominal pose instead.
        motor_targets = (
            self._default_pose
            + self._action_scale * action
        )

        motor_targets = jp.clip(
            motor_targets,
            self._ctrl_min,
            self._ctrl_max,
        )

        # Smoothness regularization is applied as a COST rather than
        # changing the action dynamics.
        action_rate_cost = (
            self._action_rate_cost_weight
            * jp.sum(jp.square(action - last_action))
        )

        # -------------------------------------------------------------
        # 2. Physics
        # -------------------------------------------------------------

        data0 = state.data

        data = mjx_env.step(
            self._mjx_model,
            data0,
            motor_targets,
            self.n_substeps,
        )

        # Advance the gait phase once per control step.  The action was
        # selected from the previous phase and is evaluated at the phase of
        # the resulting state.
        phase = self._wrap_phase(
            state.info['phase']
            + 2.0 * jp.pi * self._step_frequency * self.dt
        )

        # -------------------------------------------------------------
        # 3. Root orientation
        # -------------------------------------------------------------

        root_quat = data.qpos[3:7]
        root_quat = root_quat / jp.linalg.norm(root_quat)

        inv_quat = jp.array([
            root_quat[0],
            -root_quat[1],
            -root_quat[2],
            -root_quat[3],
        ])

        # Gravity expressed in robot coordinates.
        gravity_world = jp.array([0.0, 0.0, -1.0])

        gravity_local = math.rotate(
            gravity_world,
            inv_quat,
        )

        # 1 when upright, 0 when sideways, -1 upside-down.
        uprightness = -gravity_local[2]

        # -------------------------------------------------------------
        # 4. Translational velocity
        # -------------------------------------------------------------

        # subtree_com of the root body = whole robot COM.
        com_before = data0.subtree_com[self._body_idx]
        com_after = data.subtree_com[self._body_idx]

        velocity_world = (
            com_after - com_before
        ) / self.dt

        forward_velocity = -velocity_world[1]
        sideways_velocity = velocity_world[0]

        # Track only the commanded forward component here.  Penalizing
        # instantaneous lateral COM velocity inside this sharp exponential
        # suppresses the weight transfer needed to lift a foot.
        forward_velocity_error_squared = jp.square(
            forward_velocity - self._target_velocity
        )

        velocity_tracking_reward = (
            self._velocity_tracking_weight
            * jp.exp(
                -forward_velocity_error_squared
                / self._tracking_sigma
            )
        )

        sideways_velocity_cost = (
            self._sideways_velocity_cost_weight
            * jp.square(sideways_velocity)
            / self._tracking_sigma
        )

        # -------------------------------------------------------------
        # 5. Alternating foot trajectory and stance-foot slip
        # -------------------------------------------------------------

        desired_foot_heights = self._desired_foot_heights(phase)

        foot_positions_before = data0.site_xpos[self._feet_site_ids]
        foot_positions_after = data.site_xpos[self._feet_site_ids]

        # The sites sit at the soles.  Clamp tiny contact penetration to zero
        # before comparing against a clearance above the floor.
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

        # Do not penalize deliberate fore-aft motion of the swing foot.
        # The height schedule is exactly zero throughout the other foot's
        # stance half-cycle.
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

        # -------------------------------------------------------------
        # 6. Angular velocity / yaw
        # -------------------------------------------------------------

        # For a free joint:
        #
        # qvel[0:3] = root translation
        # qvel[3:6] = root angular velocity
        #
        # MuJoCo represents the rotational part in the local body frame.
        root_angular_velocity_local = data.qvel[3:6]

        # Rotate to world coordinates so "yaw" is specifically rotation
        # about WORLD Z.
        root_angular_velocity_world = math.rotate(
            root_angular_velocity_local,
            root_quat,
        )

        yaw_rate = root_angular_velocity_world[2]

        yaw_rate_cost = (
            self._yaw_rate_cost_weight
            * jp.square(yaw_rate)
        )

        # -------------------------------------------------------------
        # 7. Absolute heading
        # -------------------------------------------------------------

        # The robot's local forward vector is -Y.
        #
        # Rotate it into world coordinates to find which direction
        # the torso is currently pointing.
        body_forward_world = math.rotate(
            self._forward_world,
            root_quat,
        )

        body_forward_xy = body_forward_world[:2]

        # Normalize planar heading so pitch doesn't reduce its magnitude.
        heading_norm = jp.maximum(
            jp.linalg.norm(body_forward_xy),
            1e-6,
        )

        body_forward_xy_normalized = (
            body_forward_xy / heading_norm
        )

        # 1 = exactly world -Y
        # 0 = facing sideways
        # -1 = facing backward
        heading_alignment = jp.dot(
            body_forward_xy_normalized,
            self._forward_world_xy,
        )

        # Cost:
        #
        # correct direction -> 0
        # sideways          -> heading_weight
        # backward          -> 2 * heading_weight
        heading_cost = (
            self._heading_cost_weight
            * (1.0 - heading_alignment)
        )

        # -------------------------------------------------------------
        # 8. Upright orientation, height, and energy
        # -------------------------------------------------------------

        # gravity_local[:2] is zero when perfectly upright.
        tilt_cost = (
            self._orientation_cost_weight
            * jp.sum(jp.square(gravity_local[:2]))
        )

        # Vertical COM bouncing isn't useful for the task.
        vertical_velocity_cost = (
            self._vertical_velocity_cost_weight
            * jp.square(velocity_world[2])
        )

        root_z = data.qpos[2]

        base_height_cost = (
            self._base_height_cost_weight
            * jp.square(root_z - self._nominal_base_height)
            / self._base_height_tracking_sigma
        )

        energy_cost = (
            self._energy_cost_weight
            * jp.sum(
                jp.abs(data.actuator_force * data.qvel[6:])
            )
        )

        # -------------------------------------------------------------
        # 9. Health / termination
        # -------------------------------------------------------------

        min_z, max_z = self._healthy_z_range

        healthy_height = (
            (root_z >= min_z)
            & (root_z <= max_z)
        )

        healthy_orientation = (
            uprightness >= self._minimum_upright
        )

        is_healthy = (
            healthy_height
            & healthy_orientation
        )

        is_healthy_float = is_healthy.astype(jp.float32)

        healthy_reward = (
            self._healthy_reward
            * is_healthy_float
        )

        if self._terminate_when_unhealthy:
            done = 1.0 - is_healthy_float
        else:
            done = jp.array(0.0)

        # -------------------------------------------------------------
        # 10. Total reward
        # -------------------------------------------------------------

        reward = (
            velocity_tracking_reward
            + foot_phase_reward
            + healthy_reward

            - sideways_velocity_cost
            - yaw_rate_cost
            - heading_cost
            - tilt_cost
            - action_rate_cost
            - vertical_velocity_cost
            - foot_slip_cost
            - base_height_cost
            - energy_cost
        )

        # -------------------------------------------------------------
        # 11. Observation
        # -------------------------------------------------------------

        obs = self._get_obs(
            data=data,
            last_action=action,
            phase=phase,
        )

        # -------------------------------------------------------------
        # 12. Debugging metrics
        # -------------------------------------------------------------

        state.metrics.update(
            reward_velocity_tracking=velocity_tracking_reward,
            reward_foot_phase=foot_phase_reward,
            reward_alive=healthy_reward,

            cost_sideways_velocity=-sideways_velocity_cost,
            cost_yaw_rate=-yaw_rate_cost,
            cost_heading=-heading_cost,
            cost_orientation=-tilt_cost,
            cost_action_rate=-action_rate_cost,
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
            obs=obs,
            reward=reward,
            done=done,
            info={
                **state.info,
                'last_action': action,
                'phase': phase,
                'rng': step_key,
            },
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(
        self,
        data: mjx.Data,
        last_action: jp.ndarray,
        phase: jp.ndarray,
    ) -> jp.ndarray:

        # -------------------------------------------------------------
        # Orientation
        # -------------------------------------------------------------

        root_quat = data.qpos[3:7]
        root_quat = root_quat / jp.linalg.norm(root_quat)

        inv_quat = jp.array([
            root_quat[0],
            -root_quat[1],
            -root_quat[2],
            -root_quat[3],
        ])

        gravity_local = math.rotate(
            jp.array([0.0, 0.0, -1.0]),
            inv_quat,
        )

        # -------------------------------------------------------------
        # Gyroscope
        # -------------------------------------------------------------

        # MuJoCo free-joint angular velocity is already expressed in
        # the local body frame, which is exactly what an IMU gyro
        # approximately gives us.
        gyro_local = data.qvel[3:6]

        # -------------------------------------------------------------
        # Absolute heading
        # -------------------------------------------------------------

        # IMPORTANT:
        #
        # This is intentionally included during the initial
        # straight-line debugging experiment.
        #
        # It makes yaw directly observable to the policy.
        body_forward_world = math.rotate(
            self._forward_world,
            root_quat,
        )

        heading_xy = body_forward_world[:2]

        heading_norm = jp.maximum(
            jp.linalg.norm(heading_xy),
            1e-6,
        )

        heading_xy = heading_xy / heading_norm

        # -------------------------------------------------------------
        # Joints
        # -------------------------------------------------------------

        joint_positions = (
            data.qpos[7:] - self._default_pose
        )

        joint_velocities = (
            self._joint_velocity_obs_scale
            * data.qvel[6:]
        )

        # -------------------------------------------------------------
        # Periodic clock
        # -------------------------------------------------------------

        clock = jp.array([
            jp.sin(phase),
            jp.cos(phase),
        ])

        # -------------------------------------------------------------
        # Command
        # -------------------------------------------------------------

        command = jp.array([
            self._target_velocity,
        ])

        # -------------------------------------------------------------
        # Final observation
        # -------------------------------------------------------------

        obs = jp.concatenate([
            # 8 joint positions
            joint_positions,

            # 8 joint velocities
            joint_velocities,

            # 3 projected gravity
            gravity_local,

            # 3 local angular velocity
            self._gyro_obs_scale * gyro_local,

            # 2 absolute heading components
            heading_xy,

            # 8 previous actions
            last_action,

            # 2 phase clock
            clock,

            # 1 commanded speed
            command,
        ])

        return jp.clip(obs, -10.0, 10.0)

    # ------------------------------------------------------------------
    # Gait helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_phase(phase: jp.ndarray) -> jp.ndarray:
        return jp.mod(phase + jp.pi, 2.0 * jp.pi) - jp.pi

    def _desired_foot_heights(
        self,
        phase: jp.ndarray,
    ) -> jp.ndarray:
        # Left and right swing phases are exactly half a cycle apart.  The
        # positive half of sin is swing; the other half is stance.  Squaring
        # it gives zero slope at lift-off and touchdown.
        foot_phases = phase + jp.array([0.0, jp.pi])
        swing = jp.maximum(jp.sin(foot_phases), 0.0)
        return self._swing_foot_height * jp.square(swing)
