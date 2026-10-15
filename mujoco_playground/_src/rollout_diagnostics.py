"""Deterministic, non-autoresetting rollout bundles for offline gait analysis."""

import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

from etils import epath
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np


DATA_FIELDS = (
    'qpos', 'qvel', 'time', 'ctrl', 'actuator_force', 'sensordata',
    'site_xpos', 'mocap_pos', 'mocap_quat', 'xfrc_applied',
)


def collect(env, policy, seeds, steps, action_repeat=1):
  """Returns [seed, time, ...] arrays. The terminal transition is valid.

  Uses the unwrapped environment: no resets or discarded terminal states.
  After termination, state is frozen and valid=False. Observations are inputs
  to action; data, metrics, and info describe the resulting state.
  """
  if steps < 1 or action_repeat < 1 or not seeds:
    raise ValueError('Positive steps/action_repeat and at least one seed required')

  def snapshot(state):
    return {
        'data': {name: getattr(state.data, name) for name in DATA_FIELDS},
        'obs': state.obs,
        'info': jax.tree.map(lambda x: x, state.info),
    }

  def one(seed):
    key = jax.random.PRNGKey(seed)
    state = env.reset(key)
    initial = snapshot(state)

    def step(carry, unused):
      state, key, active = carry
      key, action_key = jax.random.split(key)
      obs = state.obs
      action, _ = policy(obs, action_key)
      action = jp.where(active, action, jp.zeros_like(action))

      def repeat(carry, unused):
        current, alive, reward = carry
        if getattr(current.data, 'impl', None) == mjx.Impl.WARP:
          # A vmapped cond selects every leaf, including Warp's batch-wide
          # contact buffers. Step all worlds and freeze only per-world state.
          candidate = env.step(jax.tree.map(lambda x: x, current), action)
          data = candidate.data.where(~alive, current.data)
          next_state = jax.tree.map(
              lambda new, old: jp.where(alive, new, old),
              candidate.replace(data=None), current.replace(data=None),
          ).replace(data=data)
        else:
          next_state = jax.lax.cond(
              alive, lambda s: env.step(s, action), lambda s: s, current
          )
        reward += jp.where(alive, next_state.reward, 0.0)
        alive = alive & ~next_state.done.astype(bool)
        return (next_state, alive, reward), None

      (next_state, alive, reward), _ = jax.lax.scan(
          repeat, (state, active, jp.zeros_like(state.reward)),
          None, length=action_repeat,
      )
      record = {
          **snapshot(next_state),
          'obs': obs,
          'next_obs': next_state.obs,
          'action': action,
          'reward': reward,
          'done': next_state.done,
          'valid': active,
          'metrics': next_state.metrics,
          'site_velocity': (next_state.data.site_xpos - state.data.site_xpos)
          / (env.dt * action_repeat),
      }
      return (next_state, key, alive), record

    _, trace = jax.lax.scan(
        step, (state, jax.random.fold_in(key, 1), jp.array(True)),
        None, length=steps,
    )
    return initial, trace

  return jax.device_get(jax.jit(jax.vmap(one))(jp.array(seeds, dtype=jp.uint32)))


def flatten(tree, prefix=''):
  result = {}
  if isinstance(tree, dict):
    for key, value in tree.items():
      result.update(flatten(value, f'{prefix}/{key}' if prefix else str(key)))
  else:
    array = np.asarray(tree)
    if array.dtype == object:
      raise TypeError(f'Non-numeric trace leaf: {prefix}')
    result[prefix] = array
  return result


def summarize(trace, seeds):
  summaries = []
  for i, seed in enumerate(seeds):
    valid = trace['valid'][i].astype(bool)
    metrics = {k: v[i][valid] for k, v in trace['metrics'].items()}
    item = {
        'seed': seed, 'valid_steps': int(valid.sum()),
        'terminated': bool(np.any(trace['done'][i][valid])),
        'episode_return': float(trace['reward'][i][valid].sum()),
        'mean_metrics': {k: float(v.mean()) for k, v in metrics.items()
                         if v.ndim == 1 and v.size},
    }
    for side in ('left', 'right'):
      height = metrics.get(f'{side}_foot_height')
      target = metrics.get(f'desired_{side}_foot_height')
      contact = metrics.get(f'{side}_foot_contact')
      if height is not None and target is not None and contact is not None:
        swing = target > 1e-3  # Exclude near-ground phase boundaries.
        item[side] = {
            'max_height_m': float(height.max()),
            'swing_contact_fraction': float(contact[swing].mean())
            if swing.any() else None,
            'height_rmse_m': float(np.sqrt(np.mean((height - target) ** 2))),
        }
        toe = metrics.get(f'{side}_toe_clearance')
        heel = metrics.get(f'{side}_heel_clearance')
        if toe is not None and heel is not None:
          clearance = np.minimum(toe, heel)
          item[side].update(
              max_sole_clearance_m=float(clearance.max()),
              mean_swing_sole_clearance_m=float(clearance[swing].mean())
              if swing.any() else None,
              # A fixed 1 mm threshold separates actual lift from numerical
              # contact chatter without depending on the training target.
              swing_airborne_fraction=float(
                  ((contact[swing] == 0) & (clearance[swing] > 1e-3)).mean()
              ) if swing.any() else None,
          )
    summaries.append(item)
  return summaries


def plot_trace(trace, directory, seeds):
  import matplotlib
  matplotlib.use('Agg')
  from matplotlib import pyplot as plt

  for i, seed in enumerate(seeds):
    valid = trace['valid'][i].astype(bool)
    t = trace['data']['time'][i][valid]
    metrics = {k: v[i][valid] for k, v in trace['metrics'].items()}
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for side, color in [('left', 'tab:blue'), ('right', 'tab:orange')]:
      for prefix, style in [('', '-'), ('desired_', '--')]:
        key = f'{prefix}{side}_foot_height'
        if key in metrics:
          axes[0].plot(t, 1000 * metrics[key], style, color=color, label=key)
      key = f'{side}_foot_contact'
      if key in metrics:
        axes[1].step(t, metrics[key], where='post', color=color, label=key)
      for end, style in [('toe', ':'), ('heel', '-.')]:
        key = f'{side}_{end}_clearance'
        if key in metrics:
          axes[0].plot(t, 1000 * metrics[key], style, color=color, label=key)
    for key in ['forward_velocity', 'sideways_velocity', 'yaw_rate']:
      if key in metrics:
        axes[2].plot(t, metrics[key], label=key)
    for key, values in metrics.items():
      if key.startswith(('reward_', 'cost_')):
        axes[3].plot(t, values, label=key, alpha=0.8)
    for ax, label in zip(axes, ['Height (mm)', 'Contact', 'm/s or rad/s', 'Reward terms']):
      ax.set_ylabel(label)
      ax.grid(alpha=0.25)
      if ax.lines:
        ax.legend(loc='upper left', fontsize=7, ncol=3)
    axes[-1].set_xlabel('Simulation time (s)')
    fig.suptitle(f'Deterministic rollout, seed {seed}; first episode only')
    fig.tight_layout()
    fig.savefig(directory / f'seed_{seed}.png', dpi=130)
    plt.close(fig)


def provenance(path):
  path = Path(path).resolve()
  def git(*args):
    result = subprocess.run(
        ['git', '-C', str(path if path.is_dir() else path.parent), *args],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None
  return {'path': str(path), 'commit': git('rev-parse', 'HEAD'),
          'status': git('status', '--porcelain'), 'diff': git('diff', 'HEAD')}


def model_assets(env):
  """Snapshot XML and assets, including disk-backed Rick models."""
  xml_path = Path(env.xml_path)
  try:
    assets = dict(env.model_assets)
  except NotImplementedError:
    suffixes = {'.xml', '.stl', '.obj', '.png', '.jpg', '.jpeg', '.msh', '.skn', '.bin'}
    assets = {
        p.relative_to(xml_path.parent).as_posix(): p.read_bytes()
        for p in xml_path.parent.rglob('*')
        if p.is_file() and p.suffix.lower() in suffixes and '.git' not in p.parts
    }
  assets[xml_path.name] = xml_path.read_bytes()
  # Verify the archive resolves all mesh/include references before publishing.
  mujoco.MjModel.from_xml_string(assets[xml_path.name].decode(), assets=assets)
  return xml_path, assets


def write_bundle(destination, env, initial, trace, seeds, metadata, rscope=True):
  """Writes an inspectable NPZ bundle locally or to an epath-supported URI."""
  destination = epath.Path(destination)
  with tempfile.TemporaryDirectory(prefix='rick-diagnostics-') as temporary:
    directory = Path(temporary)
    arrays = flatten({'initial': initial, 'trace': trace})
    np.savez_compressed(directory / 'traces.npz', **arrays)
    (directory / 'summary.json').write_text(json.dumps(summarize(trace, seeds), indent=2))
    plot_trace(trace, directory, seeds)
    xml_path, assets = model_assets(env)
    mujoco.mj_saveModel(env.mj_model, str(directory / 'model.mjb'))
    asset_dir = directory / 'assets'
    asset_dir.mkdir()
    for name, content in assets.items():
      relative = Path(name)
      if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe model asset path: {name}')
      target = asset_dir / relative
      target.parent.mkdir(parents=True, exist_ok=True)
      target.write_bytes(content)
    sources = directory / 'sources'
    sources.mkdir()
    paths = [Path(inspect.getfile(type(env))), Path(__file__),
             Path(sys.argv[0]), Path(inspect.getfile(type(env))).parent / 'rick_constants.py']
    for name in ['mujoco_playground._src.mjx_env', 'mujoco_playground._src.wrapper',
                 'mujoco_playground.config.locomotion_params']:
      module = sys.modules.get(name)
      if module is not None:
        paths.append(Path(inspect.getfile(module)))
    for index, source in enumerate(paths):
      if source.is_file():
        (sources / f'{index}_{source.name}').write_bytes(source.read_bytes())
    packages = {}
    for name in ['playground', 'brax', 'jax', 'jaxlib', 'mujoco', 'mujoco-mjx',
                 'warp-lang', 'rscope']:
      try:
        dist = importlib.metadata.distribution(name)
        packages[name] = {'version': dist.version,
                          'direct_url': json.loads(dist.read_text('direct_url.json') or 'null')}
      except importlib.metadata.PackageNotFoundError:
        packages[name] = None
    if rscope:
      from rscope.rollout import Rollout
      output = directory / 'rscope'
      output.mkdir()
      with (output / 'rscope_meta.pkl').open('wb') as f:
        pickle.dump({'xml_path': xml_path.name, 'model_assets': assets}, f)
      # Rscope expects [time, env, ...]. Frozen tails are marked in metrics;
      # NPZ/plots/summary use valid to exclude them entirely.
      swap = lambda x: np.swapaxes(x, 0, 1)
      priority = [
          'left_toe_clearance', 'left_heel_clearance',
          'right_toe_clearance', 'right_heel_clearance',
          'left_foot_contact', 'right_foot_contact',
          'reward_swing_airborne', 'forward_velocity',
          'left_foot_height', 'right_foot_height',
      ]
      metrics = {k: swap(trace['metrics'][k]) for k in priority if k in trace['metrics']}
      metrics.update(valid=swap(trace['valid'].astype(float)), done=swap(trace['done']))
      metrics.update(jax.tree.map(swap, trace['metrics']))
      rollout = Rollout(
          **{k: swap(trace['data'][k]) for k in
             ['qpos', 'qvel', 'mocap_pos', 'mocap_quat']},
          time=swap(initial['data']['time'][:, None] +
                    np.arange(1, trace['valid'].shape[1] + 1)[None, :]
                    * env.dt * metadata.get('ppo_config', {}).get('action_repeat', 1)),
          obs=jax.tree.map(swap, trace['next_obs']),
          reward=swap(trace['reward']), metrics=metrics,
      )
      with (output / 'rollout.mj_unroll').open('wb') as f:
        pickle.dump(rollout, f)
    manifest = {
        'schema_version': 1, **metadata, 'seeds': list(seeds),
        'ctrl_dt': env.dt, 'deterministic_policy': True,
        'layout': 'NPZ: seed,time,...; initial: seed,...; rscope: time,seed,...',
        'alignment': 'obs/action are pre-step; next_obs/data/info/metrics are post-step',
        'termination': 'valid includes terminal transition; later states frozen; no autoreset',
        'rscope_time': 'Uniform playback clock including frozen padding; NPZ retains physical time',
        'model_note': 'model.mjb is nominal; sampled episode parameters are in initial/info',
        'packages': packages,
        'environment_source': provenance(inspect.getfile(type(env))),
        'model_source': provenance(xml_path),
        'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()},
        'sha256': {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in directory.rglob('*') if p.is_file()},
    }
    (directory / 'metadata.json').write_text(json.dumps(manifest, indent=2, default=str))
    for source in directory.rglob('*'):
      if source.is_file():
        target = destination / source.relative_to(directory).as_posix()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
  return manifest


def render_videos(env, trace, destination, count, action_repeat=1):
  """Render the recorded states, ensuring videos and diagnostics correspond."""
  from types import SimpleNamespace
  import mediapy
  destination = epath.Path(destination)
  if count > trace['valid'].shape[0]:
    raise ValueError('num_videos cannot exceed the number of diagnostic seeds')
  for i in range(count):
    length = int(trace['valid'][i].sum())
    states = [SimpleNamespace(data=SimpleNamespace(**{
        key: value[i, t] for key, value in trace['data'].items()
    })) for t in range(0, length, 2)]
    frames = env.render(states, height=480, width=640)
    mediapy.write_video(
        destination / f'rollout{i}.mp4', frames,
        fps=1.0 / (2 * env.dt * action_repeat),
    )
