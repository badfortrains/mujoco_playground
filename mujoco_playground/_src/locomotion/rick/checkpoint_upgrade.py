"""Warm-start heading-aware Rick policies without changing their initial output."""

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
