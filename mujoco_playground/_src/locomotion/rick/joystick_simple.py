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
        'ctrl_dt': 0.02,
        'sim_dt': 0.002,
        'forward_reward_weight': 2.0,
        'action_rate_cost_weight': 0.4,
        'sideways_cost_weight': 0.2,
        'orientation_cost_weight': 0.2,
        'healthy_reward': 1.0,
        'terminate_when_unhealthy': True,
        'healthy_z_range': (0.05, 0.2),
        'episode_length': 1000,
        'action_repeat': 1,
    })

class Joystick(mjx_env.MjxEnv):
  def __init__(
      self, 
      task: str = "flat_terrain", 
      config: config_dict.ConfigDict = default_config(), 
      config_overrides: dict = None
  ):
    super().__init__(config, config_overrides)
    path = consts.task_to_xml(task)
    self._mj_model = mujoco.MjModel.from_xml_path(path.as_posix())
    self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    self._mj_model.opt.iterations = 10 
    self._mj_model.opt.ls_iterations = 6
    self._mjx_model = mjx.put_model(self._mj_model)

    self._step_frequency = 0.8
    self._action_dim = 8
    
    self._forward_reward_weight = config.forward_reward_weight
    self._action_rate_cost_weight = config.action_rate_cost_weight
    self._orientation_cost_weight = config.orientation_cost_weight
    self._sideways_cost_weight = config.sideways_cost_weight
    self._healthy_reward = config.healthy_reward
    self._terminate_when_unhealthy = config.terminate_when_unhealthy
    self._healthy_z_range = config.healthy_z_range
    
    self._body_idx = mujoco.mj_name2id(
        self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'body'
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
    rng, rng1, rng2, step_key, cmd_key = jax.random.split(rng, 5)
    target_velocity = jax.random.uniform(cmd_key, minval=0.0, maxval=0.12)
    
    qpos = self._mjx_model.qpos0
    
    # Normalize the quaternion
    root_quat = qpos[3:7]
    root_quat = root_quat / jp.linalg.norm(root_quat)
    qpos = qpos.at[3:7].set(root_quat)

    qvel = jp.zeros((self._mjx_model.nv,))

    data = mujoco_playground._src.mjx_env.make_data(self._mj_model, qpos=qpos, qvel=qvel)
    data = mjx.forward(self._mjx_model, data)

    last_action = jp.zeros((self._action_dim,))
    
    # Get Ground Truth Gravity in Local Frame
    inv_quat = jp.array([root_quat[0], -root_quat[1], -root_quat[2], -root_quat[3]])
    gravity_world = jp.array([0.0, 0.0, -1.0])
    gravity_local = math.rotate(gravity_world, inv_quat)
    
    obs = self._get_obs(data, gravity_local, target_velocity)
    
    reward, done, zero = jp.zeros(3)
    metrics = {
        'forward_reward': zero,
        'reward_linvel': zero,
        'reward_action_rate': zero,
        'reward_orientation': zero,
        'reward_alive': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
    }
    
    return mjx_env.State(
        data=data,
        obs=obs,
        reward=reward,
        done=done,
        metrics=metrics,
        info={
            'last_action': last_action,
            'rng': step_key,
            'target_velocity': target_velocity
        }
    )

  def step(self, state: mjx_env.State, action: jp.ndarray) -> mjx_env.State:
    rng = state.info['rng']
    rng, rng_act, rng_obs = jax.random.split(rng, 3)

    last_action = state.info['last_action']
    action_rate_cost = self._action_rate_cost_weight * jp.sum(jp.square(action - last_action))

    clipped_action = jp.clip(action, -1.0, 1.0)

    alpha = 0.3
    smoothed_action = alpha * clipped_action + (1.0 - alpha) * last_action

    ctrl_min = self._mjx_model.actuator_ctrlrange[:, 0]
    ctrl_max = self._mjx_model.actuator_ctrlrange[:, 1]
    action_scale = (ctrl_max - ctrl_min) / 2.0
    action_offset = (ctrl_max + ctrl_min) / 2.0
    scaled_action = smoothed_action * action_scale + action_offset

    data0 = state.data
    
    data = mjx_env.step(self._mjx_model, data0, scaled_action, self.n_substeps)
    
    root_quat = data.qpos[3:7]
    inv_quat = jp.array([root_quat[0], -root_quat[1], -root_quat[2], -root_quat[3]])
    gravity_world = jp.array([0.0, 0.0, -1.0])
    gravity_local = math.rotate(gravity_world, inv_quat)
    
    target_velocity = state.info['target_velocity']
    
    com_before = data0.subtree_com[self._body_idx]
    com_after = data.subtree_com[self._body_idx]
    velocity = (com_after - com_before) / self.dt
    vel_2d = velocity[:2] 
    
    forward_dir = jp.array([0.0, -1.0]) 
    forward_velocity = jp.dot(vel_2d, forward_dir)
    
    velocity_error = forward_velocity - target_velocity
    shaping_constant = 100.0 
    forward_reward = self._forward_reward_weight * jp.exp(-shaping_constant * jp.square(velocity_error))
    sideways_dir = jp.array([1.0, 0.0])

    sideways_speed = jp.dot(vel_2d, sideways_dir)
    sideways_cost = self._sideways_cost_weight * jp.abs(sideways_speed)

    projected_up = math.rotate(jp.array([0., 0., 1.]), root_quat)
    tilt_cost = self._orientation_cost_weight * jp.sum(jp.square(projected_up[:2]))

    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(data.qpos[2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(data.qpos[2] > max_z, 0.0, is_healthy)
    
    healthy_reward = self._healthy_reward if self._terminate_when_unhealthy else self._healthy_reward * is_healthy

    reward = forward_reward + healthy_reward - sideways_cost - tilt_cost - action_rate_cost
    
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    
    obs = self._get_obs(data, gravity_local, target_velocity)
    
    state.metrics.update(
        forward_reward=forward_reward,
        reward_linvel=forward_reward,
        reward_action_rate=-action_rate_cost,
        reward_orientation=-tilt_cost,
        reward_alive=healthy_reward,
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )
    
    return state.replace(
        data=data, 
        obs=obs, 
        reward=reward, 
        done=done, 
        info={
            **state.info, 
            'last_action': clipped_action,
            'rng': rng 
        }
    )

  def _get_obs(self, data: mjx.Data, gravity: jp.ndarray, target_velocity: jp.ndarray) -> jp.ndarray:
    t = data.time
    
    phase_sin = jp.sin(2.0 * jp.pi * self._step_frequency * t)
    phase_cos = jp.cos(2.0 * jp.pi * self._step_frequency * t)
    clock = jp.array([phase_sin, phase_cos])

    return jp.concatenate([
        data.qpos[7:], # joint positions
        gravity,   
        clock.flatten(),
        jp.array([target_velocity])
    ])
