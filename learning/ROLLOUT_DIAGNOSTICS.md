# Rollout diagnostics

Install the optional dependencies in the training or viewer environment:

```sh
uv pip install -e '.[diagnostics]'
```

Record a final deterministic bundle alongside a video:

```sh
train-jax-ppo --env_name=RickJoystickFlatTerrain --diagnostics --diagnostic_seeds=0,1,2,3
```

To evaluate a completed checkpoint without training:

```sh
train-jax-ppo --env_name=RickJoystickFlatTerrain --play_only \
  --load_checkpoint_path=/path/to/checkpoints --diagnostics --num_videos=1
```

New checkpoints save `training_config.json` as well as the environment
`config.json`. Play-only evaluation restores those settings before explicit
CLI overrides. Older checkpoints without the PPO configuration still use the
current PPO defaults: supply matching network settings when necessary.
The source code and assets must also match the checkpoint when reproducing an
old run; loading a checkpoint does not check out historical environment code.

`--rscope_envs=4` additionally records a bundle at each nonzero policy checkpoint
callback. `--diagnostics` records a final bundle. The per-callback recorder
replaces the disabled upstream integration and supports Rick's internal
randomization. Vision and external wrapper randomization are rejected because
the collector uses the unwrapped environment. Recording adds compilation,
evaluation, and artifact-writing overhead; final-only is the default Spot mode.

Bundles are written below the experiment's `diagnostics/` directory:

- `traces.npz`: arrays, readable with `numpy.load(..., allow_pickle=False)`.
- `summary.json`: return, termination, valid step count, mean reward terms,
  foot-height error and swing-contact fraction (desired clearance > 1 mm).
- `seed_*.png`: synchronized height, contact, velocity, and reward plots.
- `metadata.json`: schema, exact evaluated configuration, checkpoint reference,
  dependency versions/VCS origins, source provenance, and file hashes.
- `model.mjb`, `assets/`, `sources/`: nominal compiled model, XML/mesh assets,
  and snapshots of the environment/recorder/trainer sources.
- `rscope/`: native rscope metadata and trajectory files, with gait metrics
  placed first to accommodate the viewer's limited number of plots.

Arrays have `(seed, time, ...)` axes. Initial state arrays omit the time axis.
`obs` and `action` are the inputs; `data`, `next_obs`, `metrics`, and `info` are
the results. Actions are policy outputs; `data/ctrl` and the servo fields in
`info` show the commands that actually reach the simulated actuators.
`initial/info` contains sampled actuator/IMU/friction/COM parameters and RNG.
`trace/site_velocity` is displacement over the control interval, not an
instantaneous contact-point velocity. With action repeat, metrics are from the
last substep, rewards sum executed substeps, and site velocity uses the full
requested interval (including any frozen remainder after early termination).

There is no autoreset. `trace/valid` includes the terminal transition and masks
all subsequent frozen padding. Use it for every statistic. A horizon-limited
rollout with `terminated=false` was truncated by the requested recording length.
Rscope uses a uniform playback clock even for frozen padding; the NPZ retains
physical simulation time. Videos use the same trace and omit padding;
`rollout0.mp4` corresponds to the first seed.

Open a bundle directly on macOS, without overwriting rscope's active directory:

```sh
mjpython -m learning.view_diagnostics /path/to/diagnostics/final
```

On Linux use `python` instead of `mjpython`. Rscope playback uses the nominal
model; measured contacts and forces in the trace remain authoritative for
randomized episodes. Do not infer randomized contact forces from playback.

A minimal analysis:

```python
import numpy as np
with np.load('/path/to/diagnostics/final/traces.npz', allow_pickle=False) as data:
    valid = data['trace/valid'][0]
    height = data['trace/metrics/left_foot_height'][0][valid]
    target = data['trace/metrics/desired_left_foot_height'][0][valid]
    contact = data['trace/metrics/left_foot_contact'][0][valid]
    print('Peak clearance (mm):', height.max() * 1000)
    print('Swing contact fraction:', contact[target > .001].mean())
```
