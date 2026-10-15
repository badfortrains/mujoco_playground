# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a PPO agent using JAX on the specified environment."""

import datetime
import functools
import json
import os
import time
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco

from mujoco_playground import registry
from mujoco_playground import wrapper
import mujoco_playground
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

try:
  import tensorboardX
except ImportError:
  tensorboardX = None

try:
  import wandb
except ImportError:
  wandb = None


xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_PLAYGROUND_CONFIG_OVERRIDES = flags.DEFINE_string(
    "playground_config_overrides",
    None,
    "Overrides for the playground env config.",
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
)
_DIAGNOSTICS = flags.DEFINE_boolean(
    "diagnostics", False, "Save deterministic rollout data, plots, and rscope files."
)
_DIAGNOSTIC_SEEDS = flags.DEFINE_list(
    "diagnostic_seeds", ["0", "1", "2", "3"], "Fixed seeds for diagnostic rollouts."
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.3, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
)
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "Run evaluation rollouts between policy updates.",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
)
_WARP_KERNEL_CACHE_DIR = flags.DEFINE_string(
    "warp_kernel_cache_dir",
    None,
    "Directory for caching compiled Warp kernels.",
)
_LOGDIR = flags.DEFINE_string("logdir", None, "Directory for logging.")


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in mujoco_playground.manipulation._envs:
    if _VISION.value:
      return manipulation_params.brax_vision_ppo_config(env_name, _IMPL.value)
    return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.locomotion._envs:
    return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.dm_control_suite._envs:
    if _VISION.value:
      return dm_control_suite_params.brax_vision_ppo_config(
          env_name, _IMPL.value
      )
    return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)

  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def main(argv):
  """Run training and evaluation for the specified environment."""

  del argv

  if _WARP_KERNEL_CACHE_DIR.value is not None:
    import warp as wp  # pylint: disable=g-import-not-at-top

    wp.config.kernel_cache_dir = _WARP_KERNEL_CACHE_DIR.value

  diagnostics_enabled = _DIAGNOSTICS.value or bool(_RSCOPE_ENVS.value)
  if diagnostics_enabled:
    if _VISION.value or _DOMAIN_RANDOMIZATION.value:
      raise ValueError("Diagnostics currently support non-vision environments with "
                       "internal randomization (including Rick), not wrapper randomization")
    # Fail before training if an optional artifact dependency is missing.
    import matplotlib  # pylint: disable=import-outside-toplevel,unused-import
    from rscope.rollout import Rollout  # pylint: disable=import-outside-toplevel,unused-import
    from mujoco_playground._src import rollout_diagnostics
    diagnostic_seeds = [int(seed) for seed in _DIAGNOSTIC_SEEDS.value]
    if _RSCOPE_ENVS.value and not _DIAGNOSTIC_SEEDS.present:
      diagnostic_seeds = list(range(_RSCOPE_ENVS.value))
    if (not diagnostic_seeds or len(set(diagnostic_seeds)) != len(diagnostic_seeds)
        or any(seed < 0 or seed >= 2**32 for seed in diagnostic_seeds)):
      raise ValueError("Diagnostic seeds must be nonempty, unique uint32 values")
    if not _DETERMINISTIC_RSCOPE.value:
      raise ValueError("Diagnostic bundles require deterministic_rscope=True")
    if _NUM_VIDEOS.value > len(diagnostic_seeds):
      raise ValueError("num_videos cannot exceed the number of diagnostic seeds")

  # Load environment configuration
  env_cfg = registry.get_default_config(_ENV_NAME.value)

  ppo_params = get_rl_config(_ENV_NAME.value)

  # Handle checkpoint loading
  if _LOAD_CHECKPOINT_PATH.value is not None:
    # Convert to absolute path
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir() and not (ckpt_path / "commit_success.txt").exists():
      latest_ckpts = list(ckpt_path.glob("*"))
      latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir() and ckpt.name.isdigit()
                      and (ckpt / "commit_success.txt").exists()]
      latest_ckpts.sort(key=lambda x: int(x.name))
      if not latest_ckpts:
        raise ValueError(f"No complete checkpoints under {ckpt_path}")
      latest_ckpt = latest_ckpts[-1]
      restore_checkpoint_path = latest_ckpt
      print(f"Restoring from: {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint: {restore_checkpoint_path}")
  else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

  # Re-evaluation must use the trained environment/network settings.
  if _PLAY_ONLY.value and restore_checkpoint_path is not None:
    saved_dir = restore_checkpoint_path.parent
    saved_env = saved_dir / "config.json"
    if saved_env.exists():
      env_cfg = config_dict.ConfigDict(json.loads(saved_env.read_text()))
    saved_ppo = saved_dir / "training_config.json"
    if saved_ppo.exists():
      ppo_params = config_dict.ConfigDict(json.loads(saved_ppo.read_text()))
      ppo_params.num_timesteps = 0

  if _NUM_TIMESTEPS.present:
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.value:
    ppo_params.num_timesteps = 0
  if _NUM_EVALS.present:
    ppo_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    ppo_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    ppo_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    ppo_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present:
    ppo_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present:
    ppo_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present:
    ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _DISCOUNTING.present:
    ppo_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    ppo_params.learning_rate = _LEARNING_RATE.value
  if _ENTROPY_COST.present:
    ppo_params.entropy_cost = _ENTROPY_COST.value
  if _NUM_ENVS.present:
    ppo_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    ppo_params.batch_size = _BATCH_SIZE.value
  if _MAX_GRAD_NORM.present:
    ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
  if _CLIPPING_EPSILON.present:
    ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
  if _POLICY_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.policy_hidden_layer_sizes = list(
        map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
    )
  if _VALUE_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.value_hidden_layer_sizes = list(
        map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
    )
  if _POLICY_OBS_KEY.present:
    ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present:
    ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value

  env_cfg_overrides = {}
  if not _PLAY_ONLY.value or _IMPL.present:
    env_cfg_overrides["impl"] = _IMPL.value
  if _VISION.value:
    env_cfg_overrides["vision"] = True
    env_cfg_overrides["vision_config.nworld"] = ppo_params.num_envs
  if _PLAYGROUND_CONFIG_OVERRIDES.value is not None:
    env_cfg_overrides.update(json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value))

  env = registry.load(
      _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
  )
  print(f"MJX physics backend: {env.mjx_model.impl.value}", flush=True)
  if _RUN_EVALS.present:
    ppo_params.run_evals = _RUN_EVALS.value
  if _LOG_TRAINING_METRICS.present:
    ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _TRAINING_METRICS_STEPS.present:
    ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

  print(f"Environment Config:\n{env_cfg}")
  if env_cfg_overrides:
    print(f"Environment Config Overrides:\n{env_cfg_overrides}\n")
  print(f"PPO Training Parameters:\n{ppo_params}")

  # Generate unique experiment name
  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  # Set up logging directory
  logdir = epath.Path(_LOGDIR.value or "logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Initialize Weights & Biases if required
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    if wandb is None:
      raise ImportError(
          "wandb is required for --use_wandb. Install via: pip install wandb"
      )
    wandb.init(project="mjxrl", name=exp_name)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})

  # Initialize TensorBoard if required
  writer = None
  if _USE_TB.value and not _PLAY_ONLY.value and tensorboardX is not None:
    writer = tensorboardX.SummaryWriter(logdir)

  # Set up checkpoint directory
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  # Save environment configuration
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  (ckpt_path / "training_config.json").write_text(
      json.dumps(ppo_params.to_dict(), indent=2)
  )

  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  network_fn = (
      ppo_networks_vision.make_ppo_networks_vision
      if _VISION.value
      else ppo_networks.make_ppo_networks
  )
  if hasattr(ppo_params, "network_factory"):
    network_factory = functools.partial(
        network_fn, **ppo_params.network_factory
    )
  else:
    network_factory = network_fn

  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(
        _ENV_NAME.value
    )

  num_eval_envs = ppo_params.get("num_eval_envs", 128)

  if "num_eval_envs" in training_params:
    del training_params["num_eval_envs"]

  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      seed=_SEED.value,
      restore_checkpoint_path=restore_checkpoint_path,
      save_checkpoint_path=ckpt_path,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
      vision=_VISION.value,
  )

  times = [time.monotonic()]

  evaluation_history = []

  # Progress function for logging
  def progress(num_steps, metrics):
    times.append(time.monotonic())
    evaluation_history.append({
        "step": int(num_steps),
        "metrics": {key: float(value) for key, value in metrics.items()},
    })
    (logdir / "evaluation_history.json").write_text(json.dumps(evaluation_history, indent=2))

    # Log to Weights & Biases
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(metrics, step=num_steps)

    # Log to TensorBoard
    if _USE_TB.value and not _PLAY_ONLY.value and writer is not None:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
    if _RUN_EVALS.value:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
    if _LOG_TRAINING_METRICS.value:
      if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
        )

  eval_env_overrides = dict(env_cfg_overrides)
  if _VISION.value:
    eval_env_overrides["vision_config.nworld"] = num_eval_envs
  eval_env = registry.load(
      _ENV_NAME.value,
      config=config_dict.ConfigDict(env_cfg.to_dict()),
      config_overrides=eval_env_overrides,
  )

  policy_params_fn = lambda *args: None
  last_diagnostic_step = None
  diagnostic_trace = None

  def save_diagnostics(label, make_policy, policy_params, checkpoint):
    nonlocal diagnostic_trace
    policy = make_policy(policy_params, deterministic=True)
    initial, trace = rollout_diagnostics.collect(
        env, policy, diagnostic_seeds,
        (ppo_params.episode_length + ppo_params.action_repeat - 1)
        // ppo_params.action_repeat,
        ppo_params.action_repeat,
    )
    diagnostic_trace = trace
    destination = logdir / "diagnostics" / label
    rollout_diagnostics.write_bundle(
        destination, env, initial, trace, diagnostic_seeds,
        {"env_name": _ENV_NAME.value, "env_config": env_cfg.to_dict(),
         "ppo_config": ppo_params.to_dict(), "checkpoint": str(checkpoint),
         "evaluation_history": evaluation_history},
    )
    print(f"Diagnostic bundle saved: {destination}", flush=True)

  if _RSCOPE_ENVS.value:
    def policy_params_fn(current_step, make_policy, params):
      nonlocal last_diagnostic_step
      # The initial callback has no completed checkpoint to diagnose.
      if int(current_step) == 0:
        return
      last_diagnostic_step = int(current_step)
      save_diagnostics(
          f"step_{int(current_step):012d}", make_policy, params,
          ckpt_path / f"{int(current_step):012d}",
      )

  # Train or load the model
  make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
      environment=env,
      progress_fn=progress,
      policy_params_fn=policy_params_fn,
      eval_env=eval_env,
  )

  print("Done training.")
  if len(times) > 1:
    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

  if diagnostics_enabled and (last_diagnostic_step is None or _DIAGNOSTICS.value):
    candidates = sorted(
        (p for p in ckpt_path.iterdir() if p.is_dir() and p.name.isdigit()
         and (p / "commit_success.txt").exists()), key=lambda p: int(p.name)
    )
    checkpoint = restore_checkpoint_path if _PLAY_ONLY.value else (
        candidates[-1] if candidates else "in-memory policy (no saved checkpoint)"
    )
    save_diagnostics("final", make_inference_fn, params, checkpoint)

  if _NUM_VIDEOS.value == 0:
    return

  if diagnostics_enabled:
    rollout_diagnostics.render_videos(
        env, diagnostic_trace, logdir, _NUM_VIDEOS.value, ppo_params.action_repeat
    )
    return

  print("Starting inference...")

  # Create inference function.
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  infer_env_overrides = dict(env_cfg_overrides)
  if _VISION.value:
    infer_env_overrides["vision_config.nworld"] = _NUM_VIDEOS.value
  infer_env = registry.load(
      _ENV_NAME.value,
      config=config_dict.ConfigDict(env_cfg.to_dict()),
      config_overrides=infer_env_overrides,
  )

  # Run evaluation rollouts matching how training handles batched environments.
  wrapped_infer_env = wrapper.wrap_for_brax_training(
      infer_env,
      episode_length=ppo_params.episode_length,
      action_repeat=ppo_params.get("action_repeat", 1),
  )

  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(wrapped_infer_env.reset)(rng)

  empty_data = reset_states.data.__class__(
      **{k: None for k in reset_states.data.__annotations__}
  )  # pytype: disable=attribute-error
  empty_traj = reset_states.__class__(
      **{k: None for k in reset_states.__annotations__}
  )  # pytype: disable=attribute-error
  empty_traj = empty_traj.replace(data=empty_data)

  def step(carry, _):
    state, rng = carry
    rng, act_key = jax.random.split(rng)
    act_keys = jax.random.split(act_key, _NUM_VIDEOS.value)
    act = jax.vmap(jit_inference_fn)(state.obs, act_keys)[0]
    state = wrapped_infer_env.step(state, act)
    traj_data = empty_traj.tree_replace({
        "data.qpos": state.data.qpos,
        "data.qvel": state.data.qvel,
        "data.time": state.data.time,
        "data.ctrl": state.data.ctrl,
        "data.mocap_pos": state.data.mocap_pos,
        "data.mocap_quat": state.data.mocap_quat,
        "data.xfrc_applied": state.data.xfrc_applied,
    })
    return (state, rng), traj_data

  @jax.jit
  def do_rollout(state, rng):
    _, traj = jax.lax.scan(
        step, (state, rng), None, length=ppo_params.episode_length
    )
    return traj

  traj_stacked = do_rollout(reset_states, jax.random.PRNGKey(_SEED.value + 1))
  # traj_stacked has shape (time, nworld, ...), swap to (nworld, time, ...).
  traj_stacked = jax.tree.map(lambda x: jp.moveaxis(x, 0, 1), traj_stacked)
  trajectories = [None] * _NUM_VIDEOS.value
  for i in range(_NUM_VIDEOS.value):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [
        jax.tree.map(lambda x, j=j: x[j], t)
        for j in range(ppo_params.episode_length)
    ]

  # Render and save the rollout.
  render_every = 2
  fps = 1.0 / infer_env.dt / render_every
  print(f"FPS for rendering: {fps}")
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  for i, rollout in enumerate(trajectories):
    traj = rollout[::render_every]
    frames = infer_env.render(
        traj, height=480, width=640, scene_option=scene_option
    )
    media.write_video(logdir / f"rollout{i}.mp4", frames, fps=fps)
    print(f"Rollout video saved as '{logdir}/rollout{i}.mp4'.")


def run():
  """Entry point for uv/pip script."""
  app.run(main)


if __name__ == "__main__":
  run()
