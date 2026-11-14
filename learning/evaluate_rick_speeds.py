"""Evaluate a Rick checkpoint at several fixed speeds under paired conditions.

From mujoco_playground:
  .venv/bin/python -m learning.evaluate_rick_speeds /path/to/run/checkpoints
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

from brax.training import networks as brax_networks
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
import jax
from ml_collections import config_dict
import numpy as np

from mujoco_playground._src import rollout_diagnostics as diagnostics
from mujoco_playground._src.locomotion.rick import joystick


def resolve_checkpoint(path):
  path = Path(path).expanduser().resolve()
  if not (path / 'commit_success.txt').is_file():
    candidates = [p for p in path.iterdir() if p.is_dir() and p.name.isdigit()
                  and (p / 'commit_success.txt').is_file()]
    if not candidates:
      raise ValueError(f'No complete checkpoints under {path}')
    path = max(candidates, key=lambda p: int(p.name))
  return path


def load_policy(checkpoint, observation_size, action_size):
  """Restore the saved architecture, including nullable initializer settings.

  Decode the JSON explicitly: some Brax versions cannot load a saved null
  mean_kernel_init_fn through checkpoint.load_config.
  """
  config = json.loads((checkpoint / 'ppo_network_config.json').read_text())
  shape = config['observation_size']
  saved_size = shape['shape'][0] if isinstance(shape, dict) else shape
  if saved_size != observation_size or config['action_size'] != action_size:
    raise ValueError('Checkpoint observation/action layout does not match this Rick environment')
  kwargs = dict(config['network_factory_kwargs'])
  if isinstance(kwargs.get('activation'), str):
    kwargs['activation'] = brax_networks.ACTIVATION[kwargs['activation']]
  for key, value in list(kwargs.items()):
    if key.endswith('kernel_init_fn') and isinstance(value, str):
      kwargs[key] = brax_networks.KERNEL_INITIALIZER[value]
  preprocess = running_statistics.normalize if config['normalize_observations'] else lambda x, _: x
  network = ppo_networks.make_ppo_networks(
      observation_size, action_size, preprocess_observations_fn=preprocess, **kwargs)
  params = ppo_checkpoint.load(checkpoint)
  return ppo_networks.make_inference_fn(network)(params, deterministic=True)


def verify_paired_initial_conditions(initial, count, speeds):
  """Compare every initial physical state, sampled parameter, and RNG leaf."""
  for name, values in diagnostics.flatten(initial).items():
    if name in ('info/target_command', 'info/command_ticks'):
      continue
    reference = values[:count]
    for i in range(1, speeds):
      if not np.array_equal(reference, values[i * count:(i + 1) * count]):
        raise ValueError(f'Initial conditions differ across speeds: {name}')


def score(trace, seeds, speeds, warmup):
  rows = []
  for j, speed in enumerate(speeds):
    for k, seed in enumerate(seeds):
      i = j * len(seeds) + k
      valid = trace['valid'][i].astype(bool)
      settled = valid & (trace['data']['time'][i] >= warmup)
      metrics = {name: values[i] for name, values in trace['metrics'].items()}
      ready = settled & (metrics['cycle_window_ready'] > 0)
      mean = lambda values, mask: float(np.mean(values[mask])) if mask.any() else None
      actual = mean(metrics['forward_velocity'] * 1000, settled)
      row = {
          'seed': seed, 'command_mm_s': float(speed),
          'actual_mm_s': actual,
          'speed_error_mm_s': actual - speed if actual is not None else None,
          'cycle_mae_mm_s': mean(abs(metrics['cycle_forward_velocity_error']) * 1000, ready),
          'mean_abs_heading_deg': mean(abs(metrics['heading_error']) * 180 / np.pi, settled),
          'duration_s': float(trace['data']['time'][i][valid][-1]),
          'terminated': bool(np.any(trace['done'][i][valid])),
      }
      for side in ('left', 'right'):
        swing = settled & (metrics[f'desired_{side}_foot_height'] > .001)
        clearance = np.minimum(metrics[f'{side}_toe_clearance'], metrics[f'{side}_heel_clearance'])
        airborne = (metrics[f'{side}_foot_contact'] == 0) & (clearance > .001)
        fraction = mean(airborne.astype(float) * 100, swing)
        row[f'{side}_swing_airborne_pct'] = fraction
      rows.append(row)
  return rows


def summarize_response(rows, seeds):
  response = []
  for seed in seeds:
    points = sorted((r for r in rows if r['seed'] == seed), key=lambda r: r['command_mm_s'])
    complete = all(not p['terminated'] and p['actual_mm_s'] is not None for p in points)
    response.append({
        'seed': seed, 'all_episodes_survived': complete,
        'requested_increase_mm_s': points[-1]['command_mm_s'] - points[0]['command_mm_s'],
        'actual_increase_mm_s': points[-1]['actual_mm_s'] - points[0]['actual_mm_s'] if complete else None,
    })
  return response


def plot_comparison(rows, seeds, destination):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(math.ceil(len(seeds) / 2), min(len(seeds), 2),
                           figsize=(9, 3.5 * math.ceil(len(seeds) / 2)), squeeze=False)
  commands = [r['command_mm_s'] for r in rows]
  lo, hi = min(commands), max(commands)
  for axis, seed in zip(axes.flat, seeds):
    points = sorted((r for r in rows if r['seed'] == seed), key=lambda r: r['command_mm_s'])
    axis.plot([lo, hi], [lo, hi], '--', color='0.5', label='Exact tracking')
    axis.plot([r['command_mm_s'] for r in points], [r['actual_mm_s'] for r in points],
              'o-', label='Policy')
    axis.set(title=f'Seed {seed}', xlabel='Command (mm/s)', ylabel='Mean speed after startup (mm/s)')
    axis.grid(alpha=.25)
    axis.legend()
  for axis in list(axes.flat)[len(seeds):]:
    axis.set_visible(False)
  fig.tight_layout()
  fig.savefig(destination, dpi=150)
  plt.close(fig)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('checkpoint', type=Path, help='Numbered checkpoint or its checkpoints/ parent')
  parser.add_argument('--speeds-mm-s', nargs='+', type=float, default=[30., 45., 60.])
  parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3])
  parser.add_argument('--seconds', type=float, default=20.)
  parser.add_argument('--warmup-seconds', type=float, default=2.)
  parser.add_argument('--impl', choices=['jax', 'warp'], default='jax')
  parser.add_argument('--rscope', action='store_true', help='Also export rscope playback (requires rscope)')
  parser.add_argument('--xml-path', type=Path, help='Override the model saved with diagnostics')
  parser.add_argument('--output', type=Path)
  args = parser.parse_args(argv)
  if args.rscope:
    from rscope.rollout import Rollout  # pylint: disable=unused-import
  if (len(args.speeds_mm_s) < 2 or len(set(args.speeds_mm_s)) != len(args.speeds_mm_s)
      or not np.all(np.isfinite(args.speeds_mm_s))):
    parser.error('Specify at least two different finite speeds')
  if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
    parser.error('Seeds must be unique uint32 values')
  if (not np.isfinite(args.seconds) or not np.isfinite(args.warmup_seconds)
      or not 0 <= args.warmup_seconds < args.seconds):
    parser.error('Require 0 <= warmup-seconds < seconds')
  checkpoint = resolve_checkpoint(args.checkpoint)
  run = checkpoint.parent.parent
  saved = json.loads((checkpoint.parent / 'config.json').read_text())
  if saved.get('motion_command_version') != 1:
    parser.error('Evaluate a trained speed/steering checkpoint with motion_command_version=1')
  bounds = saved['forward_velocity_range']
  if any(not bounds[0] - 1e-8 <= v / 1000 <= bounds[1] + 1e-8 for v in args.speeds_mm_s):
    parser.error(f'Requested speeds exceed the policy limits: {bounds} m/s')
  bundle = run / 'diagnostics/final'
  model = args.xml_path
  if model is None:
    metadata = json.loads((bundle / 'metadata.json').read_text())
    model = bundle / 'assets' / Path(metadata['model_source']['path']).name
    expected = metadata['sha256'].get('assets/' + model.name)
    if expected and hashlib.sha256(model.read_bytes()).hexdigest() != expected:
      raise ValueError('Saved model XML checksum does not match its diagnostic metadata')
  model = model.expanduser().resolve()
  if not model.is_file():
    parser.error(f'Model XML not found: {model}; pass --xml-path if needed')
  cfg = config_dict.ConfigDict(saved)
  cfg.impl = args.impl
  cfg.command_resample_in_episode = False
  cfg.command_alternate_speed_endpoints = False
  cfg.command_speed_levels = ()
  env = joystick.Joystick(config=cfg, xml_path=model)
  steps = math.ceil(args.seconds / env.dt)
  output = (args.output or run / ('speed_sweep-' + datetime.datetime.now(
      datetime.timezone.utc).strftime('%Y%m%d-%H%M%S'))).expanduser().resolve()
  output.mkdir(parents=True, exist_ok=False)
  os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
  policy = load_policy(checkpoint, env.observation_size, env.action_size)
  seeds = args.seeds * len(args.speeds_mm_s)
  commands = [[v / 1000, 0.] for v in args.speeds_mm_s for _ in args.seeds]
  print(f'Checkpoint: {checkpoint}\nOutput: {output}', flush=True)
  print(f'Compiling and evaluating {len(seeds)} paired episodes ({args.seconds:g} s each)...', flush=True)
  initial, trace = diagnostics.collect(env, policy, seeds, steps, target_commands=commands)
  verify_paired_initial_conditions(initial, len(args.seeds), len(args.speeds_mm_s))
  rows = score(trace, args.seeds, args.speeds_mm_s, args.warmup_seconds)
  response = summarize_response(rows, args.seeds)
  with (output / 'comparison.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  summary = {'checkpoint': str(checkpoint), 'model_xml': str(model), 'impl': args.impl,
             'speeds_mm_s': args.speeds_mm_s, 'seeds': args.seeds,
             'warmup_seconds': args.warmup_seconds, 'matched_initial_conditions': True,
             'episodes': rows, 'response': response}
  (output / 'comparison.json').write_text(json.dumps(summary, indent=2))
  plot_comparison(rows, args.seeds, output / 'comparison.png')
  for j, speed in enumerate(args.speeds_mm_s):
    subset = slice(j * len(args.seeds), (j + 1) * len(args.seeds))
    diagnostics.write_bundle(
        output / f'speed_{speed:g}_mm_s', env,
        jax.tree.map(lambda x: x[subset], initial),
        jax.tree.map(lambda x: x[subset], trace), args.seeds,
        {'env_name': 'RickJoystickFlatTerrain', 'env_config': cfg.to_dict(),
         'checkpoint': str(checkpoint), 'evaluation_target_command': [speed / 1000, 0.],
         'matched_initial_conditions': True}, rscope=args.rscope)
  print('\nSeed  Command  Actual  Error  Cycle MAE  Fell', flush=True)
  for r in rows:
    number = lambda x: f'{x:6.1f}' if x is not None else '   n/a'
    print(f"{r['seed']:4} {r['command_mm_s']:8.1f} {number(r['actual_mm_s'])} "
          f"{number(r['speed_error_mm_s'])} {number(r['cycle_mae_mm_s'])}  {r['terminated']}")
  print('\nSpeed response (ideal increase equals requested increase):')
  for r in response:
    print(r)
  print(f'\nComparison and per-speed diagnostics saved: {output}', flush=True)


if __name__ == '__main__':
  main()
