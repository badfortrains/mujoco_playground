"""Warm-start Rick policies while preserving behavior at trained commands."""

import copy

from jax import numpy as jp
import numpy as np


def add_heading_inputs(params, old_obs_dim):
    """Append two zero-weight inputs to policy/value and unit normalizer stats.

    Accepts parameters returned by Brax PPO checkpoint.load. Existing inputs,
    weights and normalization are retained; new heading inputs initially have
    no influence. The checkpoint on disk is never modified.
    """
    normalizer, policy, value = params
    if not hasattr(normalizer.mean, 'shape') or normalizer.mean.shape != (old_obs_dim,):
        raise ValueError('Heading upgrade requires a flat Rick observation normalizer')
    if hasattr(normalizer.count, 'hi'):
        count = (int(np.asarray(normalizer.count.hi)) * 2**32
                 + int(np.asarray(normalizer.count.lo)))
    else:
        count = float(np.asarray(normalizer.count))
    def extend(array, fill):
        return jp.concatenate([jp.asarray(array), jp.full(2, fill, dtype=array.dtype)])
    normalizer = normalizer.replace(
        mean=extend(normalizer.mean, 0.0),
        std=extend(normalizer.std, 1.0),
        summed_variance=extend(normalizer.summed_variance, count),
    )
    networks = []
    for network in (policy, value):
        network = copy.deepcopy(network)
        kernel = network['params']['hidden_0']['kernel']
        if kernel.shape[0] != old_obs_dim:
            raise ValueError('Heading upgrade requires policy/value to use the state observation')
        network['params']['hidden_0']['kernel'] = jp.concatenate(
            [jp.asarray(kernel), jp.zeros((2, kernel.shape[1]), dtype=kernel.dtype)], axis=0
        )
        networks.append(network)
    return (normalizer, *networks)


def add_motion_command_input(params, speed_index, reference_speed,
                             speed_std=0.05, yaw_std=0.3):
    """Convert [history, IMU, clock, speed, heading] to speed/yaw/heading.

    A fixed-speed policy never learned a speed response: that input can have
    std=1e-6 and an arbitrary kernel row. Fold its contribution at the trained
    speed into the first-layer bias, then initialize both command rows to zero.
    Thus all new speed/yaw commands initially reproduce the old walking policy
    at reference_speed, including during the START ramp from zero.

    Preserve the other normalization statistics and count. Seed finite speed
    and yaw variances consistent with that count so the first statistics update
    does not reinstate the old near-zero speed scale. Source params are untouched.
    """
    normalizer, policy, value = params
    old_dim = speed_index + 3  # Speed followed by sin/cos heading.
    if (not hasattr(normalizer.mean, 'shape')
            or normalizer.mean.shape != (old_dim,)
            or normalizer.std.shape != (old_dim,)
            or normalizer.summed_variance.shape != (old_dim,)):
        raise ValueError('Motion upgrade requires the flat heading-aware Rick layout')
    if (not np.isfinite(reference_speed) or not np.isfinite(speed_std)
            or not np.isfinite(yaw_std) or speed_std <= 0 or yaw_std <= 0):
        raise ValueError('Motion upgrade requires finite reference speed and positive scales')
    # Brax Welford stores an accumulated variance; EMA has different semantics.
    if getattr(normalizer, 'mode', 0) != 0 or getattr(normalizer, 'std_eps', 0.0) != 0:
        raise ValueError('Motion upgrade requires Welford normalization with std_eps=0')
    if hasattr(normalizer.count, 'hi'):
        count = int(np.asarray(normalizer.count.hi)) * 2**32 + int(np.asarray(normalizer.count.lo))
    else:
        count = float(np.asarray(normalizer.count))
    if not np.isfinite(count) or count <= 0:
        raise ValueError('Motion upgrade requires nonempty normalization statistics')
    old_speed_std = float(np.asarray(normalizer.std[speed_index]))
    if not np.isfinite(old_speed_std) or old_speed_std <= 0:
        raise ValueError('Invalid saved speed normalization scale')
    constant_speed = (jp.asarray(reference_speed, dtype=normalizer.mean.dtype)
                      - normalizer.mean[speed_index]) / normalizer.std[speed_index]

    def replace_commands(array, speed, yaw):
        return jp.concatenate([
            jp.asarray(array[:speed_index]),
            jp.asarray([speed, yaw], dtype=array.dtype),
            jp.asarray(array[speed_index + 1:]),
        ])

    normalizer = normalizer.replace(
        mean=replace_commands(normalizer.mean, reference_speed, 0.0),
        std=replace_commands(normalizer.std, speed_std, yaw_std),
        summed_variance=replace_commands(normalizer.summed_variance,
                                         count * speed_std**2, count * yaw_std**2),
    )
    networks = []
    for network in (policy, value):
        network = copy.deepcopy(network)
        layer = network['params']['hidden_0']
        kernel = jp.asarray(layer['kernel'])
        if kernel.shape[0] != old_dim:
            raise ValueError('Motion upgrade requires policy/value to use the state observation')
        layer['bias'] = jp.asarray(layer['bias']) + constant_speed * kernel[speed_index]
        layer['kernel'] = jp.concatenate([
            kernel[:speed_index], jp.zeros((2, kernel.shape[1]), dtype=kernel.dtype),
            kernel[speed_index + 1:],
        ], axis=0)
        networks.append(network)
    return (normalizer, *networks)


def validate_walking_upgrade(saved_config, new_config, old_obs_dim, action_dim):
    """Reject unrelated checkpoints before inserting the yaw command input."""
    if (saved_config.get('motion_command_version', 0) != 0
            or new_config.get('motion_command_version') != 1
            or not saved_config.get('heading_observation', False)
            or not new_config.get('heading_observation', False)):
        raise ValueError('Warm start requires a fixed-speed heading-aware Rick checkpoint')
    for name in ('command_history_length', 'ctrl_dt', 'action_scale', 'gyro_obs_scale',
                 'gyro_bias_calibration', 'step_frequency'):
        old, new = saved_config.get(name), new_config.get(name)
        if old is None or new is None or not np.isclose(old, new):
            raise ValueError(f'Walking checkpoint has incompatible {name}: {old} vs {new}')
    speed_index = saved_config['command_history_length'] * action_dim + 8
    if old_obs_dim != speed_index + 3:
        raise ValueError('Walking checkpoint has incompatible observation layout')
    return speed_index


def activate_turn_command(params, yaw_index, yaw_std=0.3):
    """Enable turning after a straight-only stage, preserving output at yaw=0.

    A constant yaw input has no learned response and its variance can collapse.
    Seed its statistics and zero its first-layer row before varying the command.
    Call only when the saved command range was exactly zero.
    """
    normalizer, policy, value = params
    if (normalizer.mean.shape != (yaw_index + 3,)
            or getattr(normalizer, 'mode', 0) != 0
            or getattr(normalizer, 'std_eps', 0.0) != 0):
        raise ValueError('Turn activation requires flat Rick Welford statistics')
    if not np.isfinite(yaw_std) or yaw_std <= 0:
        raise ValueError('Turn activation requires a positive finite scale')
    if hasattr(normalizer.count, 'hi'):
        count = int(np.asarray(normalizer.count.hi)) * 2**32 + int(np.asarray(normalizer.count.lo))
    else:
        count = float(np.asarray(normalizer.count))
    old_std = float(np.asarray(normalizer.std[yaw_index]))
    if not np.isfinite(count) or count <= 0 or not np.isfinite(old_std) or old_std <= 0:
        raise ValueError('Turn activation requires valid saved statistics')
    constant_yaw = -normalizer.mean[yaw_index] / normalizer.std[yaw_index]
    normalizer = normalizer.replace(
        mean=jp.asarray(normalizer.mean).at[yaw_index].set(0.),
        std=jp.asarray(normalizer.std).at[yaw_index].set(yaw_std),
        summed_variance=jp.asarray(normalizer.summed_variance).at[yaw_index].set(count * yaw_std**2),
    )
    networks = []
    for network in (policy, value):
        network = copy.deepcopy(network)
        layer = network['params']['hidden_0']
        kernel = jp.asarray(layer['kernel'])
        if kernel.shape[0] != yaw_index + 3:
            raise ValueError('Turn activation requires the Rick observation layout')
        layer['bias'] = jp.asarray(layer['bias']) + constant_yaw * kernel[yaw_index]
        layer['kernel'] = kernel.at[yaw_index].set(0.)
        networks.append(network)
    return (normalizer, *networks)
