"""Check diagnostic playback integration without training or creating a GPU VM."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from absl import flags
from absl.testing import flagsaver
from ml_collections import config_dict

from learning import train_jax_ppo as trainer
from mujoco_playground._src import rollout_diagnostics


class PlaybackDiagnosticsTest(unittest.TestCase):
  def test_playback_restores_config_and_exports_matching_checkpoint(self):
    if not flags.FLAGS.is_parsed():
      flags.FLAGS(['test'])
    with tempfile.TemporaryDirectory() as root:
      root = Path(root)
      checkpoint = root / 'checkpoints/000000000123'
      checkpoint.mkdir(parents=True)
      (checkpoint / 'commit_success.txt').write_text('ok')
      (checkpoint.parent / 'config.json').write_text(json.dumps({
          'impl': 'jax', 'episode_length': 17, 'saved_marker': 42,
      }))
      (checkpoint.parent / 'training_config.json').write_text(json.dumps({
          'num_timesteps': 100000, 'episode_length': 17, 'action_repeat': 1,
          'num_envs': 8, 'network_factory': {'policy_hidden_layer_sizes': [32, 16]},
      }))
      make_policy = mock.Mock(return_value=object())
      train = mock.Mock(return_value=(make_policy, 'parameters', {}))
      env = object()
      with flagsaver.flagsaver(
          env_name='RickJoystickFlatTerrain', play_only=True, diagnostics=True,
          diagnostic_seeds=['0', '2'], rscope_envs=None, vision=False,
          domain_randomization=False, num_videos=0,
          load_checkpoint_path=str(checkpoint.parent), logdir=str(root / 'output'),
      ), mock.patch.object(trainer.registry, 'get_default_config', return_value=config_dict.ConfigDict()), mock.patch.object(
          trainer, 'get_rl_config', return_value=config_dict.ConfigDict()
      ), mock.patch.object(trainer.registry, 'load', return_value=env) as load, mock.patch.object(
          trainer.ppo, 'train', train
      ), mock.patch.object(rollout_diagnostics, 'collect', return_value=('initial', 'trace')) as collect, mock.patch.object(
          rollout_diagnostics, 'write_bundle'
      ) as write:
        trainer.main([])
      self.assertEqual(train.call_args.kwargs['num_timesteps'], 0)
      self.assertEqual(train.call_args.kwargs['episode_length'], 17)
      self.assertEqual(load.call_args_list[0].kwargs['config'].saved_marker, 42)
      self.assertEqual(train.call_args.kwargs['network_factory'].keywords['policy_hidden_layer_sizes'], [32, 16])
      collect.assert_called_once_with(env, make_policy.return_value, [0, 2], 17, 1)
      self.assertEqual(write.call_args.args[5]['checkpoint'], str(checkpoint))
      self.assertEqual(write.call_args.args[5]['env_config']['saved_marker'], 42)
      make_policy.assert_called_once_with('parameters', deterministic=True)


if __name__ == '__main__':
  unittest.main()
