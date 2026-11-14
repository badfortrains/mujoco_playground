# Recover walking, then widen commands

`RickJoystickFlatTerrain` now defaults to the `walk` stage. Cadence remains
fixed at the model-derived frequency (about 1.415 Hz). All stages use the same
108-input observation and deployment command protocol.

| `--rick_command_stage` | Sampled forward speed | Yaw rate | Command changes |
| --- | --- | --- | --- |
| `walk` (default) | 50 mm/s | 0 | One target per episode |
| `speed` | 30–60 mm/s | 0 | One target per episode |
| `transitions` | 30–60 mm/s; 15% standing | 0 | Every 3–6 s |
| `speed_jumps` (experiment) | Alternate 30 and 60 mm/s | 0 | Every 4–6 s; strict cycle-speed objective |
| `speed_levels` (experiment) | Cycle 30, 45, 60 mm/s | 0 | Every 4–6 s; strict cycle-speed objective |
| `turns` | 30–60 mm/s; no standing | ±0.15 rad/s | Every 4–6 s; 50% straight / 50% turning; strict speed tracking |
| `full` | −40 to +80 mm/s | ±0.60 rad/s | Every 3–6 s; standing, pivot, straight, combined |

Every episode starts with zero commanded motion and ramps toward its target.
Speed and yaw acceleration limits remain 50 mm/s² and 1 rad/s². Stages advance
explicitly between runs; they do not advance automatically with training steps.

## Start from the successful walking policy

From the `mujoco_playground` repository:

```sh
.venv/bin/python learning/train_jax_ppo.py \
  --env_name=RickJoystickFlatTerrain \
  --rick_command_stage=walk \
  --load_checkpoint_path=/Users/suzanna/Documents/code/rick_pico/checkpoints/20261113-034820/checkpoints \
  --diagnostics
```

Keep your usual PPO budget and hardware flags. For the existing cloud workflow,
`rick_spot/automate_training.py` accepts both `--rick-command-stage` and
`--load-checkpoint-path` before `--`:

```sh
./automate_training.py \
  --rick-command-stage=walk \
  --load-checkpoint-path=../rick_pico/checkpoints/20261113-034820/checkpoints \
  -- --num_timesteps=40000000 --num_evals=9
```

Run this from `rick_spot` with your usual cloud options. It uploads the selected
local checkpoint, its saved environment config, and the modified training
sources. A checkpoint parent directory selects its latest complete checkpoint.
Final diagnostics are generated automatically. The older passthrough spelling
`-- --rick_command_stage=walk` remains supported.

The trainer converts a compatible, heading-aware 107-input walking checkpoint
in memory: it inserts yaw before heading, preserves actor/value output at the
old trained speed, and initializes both command input rows to zero. It seeds
speed/yaw normalization variances to avoid the old constant-speed input's tiny
scale. The saved checkpoint is not changed. This is a warm start with a fresh
optimizer, not an exact continuation of the old objective. The action scale,
control interval, history layout, gyro settings, and cadence must match.

For each later stage, restore the successful checkpoint from the preceding
stage and change `--rick_command_stage`. When yaw is first enabled after a
straight-only stage, its unused input row and normalization are initialized
while preserving zero-yaw behavior and the learned speed response. Restoring a
command policy without a stage flag uses the new `walk` default for training;
choose the desired stage explicitly. `--play_only` uses saved settings by
default. Historical returns require the run's saved source, since rewards changed.

## Reward changes

Speed reward retains total weight 2. Once one gait period is recorded, 75% uses
cycle-average forward speed error with sigma 0.0004, and 25% uses instantaneous
error with sigma 0.0025. Each historical sample is compared with its own command
before averaging. Before a complete cycle, the broad instantaneous term gets the
full weight. This tolerates periodic surges while still penalizing sustained
speed error. New metrics expose both reward terms and cycle speed error.

Swing progress has weight 1 and pays only when the swing foot is off the ground,
the opposite foot has contact, and both toe and heel clear the floor. Clearance
credit rises smoothly to full credit at 5 mm, then saturates. Placement and
stance-retraction bonuses have been removed. Dense lift/unloading rewards remain
independent so the policy can discover liftoff before the progress gate opens.

## When to advance

Review deterministic rollouts before widening the task. In `walk`, look for
repeated supported steps, toe **and** heel clearance, sustained forward travel
near 50 mm/s over complete cycles, and full episodes without tilt termination.
Total return alone is insufficient, and returns across the old/new rewards are
not directly comparable. In `speed`, inspect both endpoints; in `transitions`,
inspect stopping and restarting; in `turns`, inspect both turn signs. Enable
reverse and pivot training only after these behaviors hold up.

For custom stages, `--playground_config_overrides` JSON overrides the selected
preset. `command_forward_velocity_range` controls sampling;
`forward_velocity_range` controls exported limits and must include zero and the
sampled range. `command_mode_probabilities` gives standing, pivot, straight,
combined probabilities, and `command_resample_in_episode` controls resampling.
Firmware/UI ranges follow the exported stage limits. Early stages are for gait
recovery and have not learned the full command range.

## Compare fixed speeds under identical conditions

Run the local evaluation from `mujoco_playground`:

```sh
.venv/bin/python -m learning.evaluate_rick_speeds \
  ../rick_pico/checkpoints/20261113-162335/checkpoints \
  --speeds-mm-s 30 45 60 \
  --seeds 0 1 2 3
```

This loads the latest complete checkpoint and runs twelve deterministic,
20-second episodes: every combination of speed and seed. Reusing a seed
preserves the initial pose, gait phase, servo properties, friction, sensor
noise sequence, and yaw-disturbance schedule. The evaluator verifies that all
initial physical states, sampled parameters, and RNG keys match across speeds.
Only the target speed changes. Commands retain the normal START ramp; targets
are held throughout each episode. Speed statistics exclude the first two
seconds, and terminal padding is excluded after a fall.

Outputs go into a new `speed_sweep-<timestamp>` directory beside `checkpoints/`:

- `comparison.csv` and `comparison.json`: per-seed command/actual speed, signed
  error, cycle-average absolute error, heading error, clearance/contact fraction,
  and termination status. JSON also reports each seed's increase in actual speed
  between the lowest and highest targets.
- `comparison.png`: actual versus requested speed, with one panel per seed.
- `speed_30_mm_s/`, `speed_45_mm_s/`, `speed_60_mm_s/`: full diagnostic bundles
  with NPZ traces, summaries, plots, model assets, and source/package provenance.

A responsive policy should increase its realized speed as the requested speed
increases. A line near the diagonal indicates accurate tracking; a nearly flat
line indicates weak speed response, even if it stays upright. Examine each seed,
especially seed 2, rather than only the average across seeds. Falls invalidate
that seed's overall response comparison and remain visible in the episode rows.

The evaluator uses the XML and meshes saved with the original diagnostics and
the saved network architecture/normalization. It defaults to the local JAX
backend; GPU installations can use `--impl=warp`. Package versions and backend
may differ from the cloud run, so compare the three speeds within this sweep.
It evaluates the policy directly and does not start PPO training or export firmware.

Use `--seeds 2` to inspect just the weak case, or `--seconds 10` for shorter
rollouts. Pass `--output /path/to/new-directory` to choose the output location.
If rscope is installed, `--rscope` also exports playback bundles:

```sh
mjpython -m learning.view_diagnostics /path/to/sweep/speed_45_mm_s
```


## Speed-jump experiment

`speed_jumps` tests whether forcing large changes and increasing the tracking
incentive fixes the weak command response seen in the first transitions run.
It starts randomly at 30 or 60 mm/s, then always switches to the opposite
endpoint every 4–6 seconds. Standing and commanded turning are disabled. The
same sampled robot remains in each episode, and ordinary acceleration limits
still smooth each change. Cadence and all gait-shaping terms are retained.

The cycle speed sigma is 0.0001 (exp(-1) at 10 mm/s error), versus 0.0004
previously. Total velocity tracking weight is 4 instead of 2; the 75% cycle /
25% broad instantaneous split is retained. At the same 15 mm/s instantaneous
and cycle error, an unresponsive 45 mm/s walker earns about 31% of perfect
tracking credit, versus 66% under the previous objective. This is an experiment,
not an automatic promotion to steering or reverse.

From `rick_spot`:

```sh
./automate_training.py \
  --rick-command-stage=speed_jumps --zone=asia-east1-b \
  --load-checkpoint-path=../rick_pico/checkpoints/20261113-180323/checkpoints \
  -- --num_timesteps=40000000 --num_evals=9
```

Compare the initial and final evaluations within this run, since its return
scale differs from previous stages. Inspect realized speeds at both endpoints,
fall frequency, and sole clearance. The fixed-speed evaluator disables
alternation so it can still test 30, 45 and 60 mm/s independently on a trained
`speed_jumps` policy.


## Three-level speed experiment

`speed_levels` adds explicit 45 mm/s training after the endpoint-trained policy
showed a weak midpoint response in seed 1. Every episode starts at a randomly
selected member of (30, 45, 60) mm/s, then cycles through that ordered list,
wrapping 60 back to 30. Each target lasts 4–6 seconds, so a full 20-second
episode encounters all three speeds. The normal acceleration limits apply.
Standing and commanded yaw remain disabled.

It retains the `speed_jumps` reward settings: velocity tracking weight 4,
cycle sigma 0.0001, 75% cycle-average / 25% broad instantaneous tracking.
Cadence, observations, and gait rewards are unchanged. This isolates exposure
to the intermediate target from another reward or control change.

```sh
./automate_training.py \
  --rick-command-stage=speed_levels --zone=asia-east1-b \
  --load-checkpoint-path=../rick_pico/checkpoints/20261113-184906/checkpoints \
  -- --num_timesteps=40000000 --num_evals=9
```

Afterward, run the fixed 30/45/60 sweep and compare with
`20261113-184906/speed_sweep`. In particular, check seed 1 at 45 mm/s and ensure
the other midpoint and endpoint responses and foot clearance are retained.
The evaluator disables the training sequence and holds each requested speed.

## Gentle turning from the three-level checkpoint

The `turns` stage samples forward speeds uniformly from 30–60 mm/s, with
half the commands straight and half combined forward/turn motion at yaw rates
uniformly sampled from ±0.15 rad/s. Targets change every 4–6 seconds. Standing,
reverse, and pivot commands remain disabled. It retains the stronger speed
objective from `speed_levels`: weight 4, cycle sigma 0.0001, and 75% cycle /
25% instantaneous tracking. Cadence and gait rewards are unchanged.

```sh
./automate_training.py \
  --rick-command-stage=turns --zone=asia-east1-b \
  --load-checkpoint-path=../rick_pico/checkpoints/20261113-195629/checkpoints \
  -- --num_timesteps=40000000 --num_evals=9
```

Check both turn signs, actual versus requested yaw rate, heading error, survival,
and swing clearance. Repeat the matched straight 30/45/60 sweep against
`20261113-195629/speed_sweep` to detect loss of speed control.

### Focused turn-tracking follow-up

The first gentle-turn run (`20261113-210350`) preserved walking in its four
cloud diagnostic episodes, but some settled turns substantially undershot the
requested rate. Its cycle-average yaw cost has weight 4: a persistent 0.1 rad/s
error costs only 0.04 per step. A focused follow-up increases that weight to
100, making the same error cost 1.0 per step. The cycle filter averages a full
gait period before squaring, so alternating body yaw can cancel normally.
The instantaneous yaw penalty, heading penalty, stronger speed objective,
command distribution, cadence, and gait rewards stay unchanged.

```sh
./automate_training.py \
  --rick-command-stage=turns --zone=asia-east1-b \
  --load-checkpoint-path=../rick_pico/checkpoints/20261113-195629/checkpoints \
  -- --num_timesteps=40000000 --num_evals=9 \
  --playground_config_overrides='{"cycle_yaw_rate_cost_weight":100.0}'
```

The follow-up returns to `195629`, the same speed-control checkpoint used to
start the first turn run. Matched straight-speed tests found that the first
turn run increased mean absolute speed bias from 3.66 to 5.71 mm/s, despite
all 12 episodes surviving and remaining monotonic. Using the same initial
checkpoint also makes this a comparison of the turn-cost change.

The explicit override is saved with the new run. It leaves the ordinary
`turns` preset unchanged. Compare initial and final evaluations within the new
run because the reward scale changes. Check actual turn rates in both
directions, cycle-average yaw error, survival, and foot clearance, then repeat
the matched straight-speed sweep. A higher return alone does not establish
better command tracking.
