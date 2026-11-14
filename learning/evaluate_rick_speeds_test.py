"""Paired speed comparisons must detect changed conditions and exclude padding."""
import unittest

import numpy as np

from learning import evaluate_rick_speeds as sweep


class SpeedSweepTest(unittest.TestCase):
  def test_pair_verification_checks_physics_parameters_and_rng(self):
    initial = {'data': {'qpos': np.array([[1.], [2.], [1.], [2.]])},
               'info': {'friction': np.array([.3, .5, .3, .5]),
                        'rng': np.array([[1, 2], [3, 4], [1, 2], [3, 4]]),
                        'target_command': np.array([[.03, 0.]] * 2 + [[.06, 0.]] * 2)}}
    sweep.verify_paired_initial_conditions(initial, 2, 2)
    initial['info']['friction'][2] = .4
    with self.assertRaisesRegex(ValueError, 'friction'):
      sweep.verify_paired_initial_conditions(initial, 2, 2)
    initial['info']['friction'][2] = .3
    initial['info']['rng'][2, 0] = 9
    with self.assertRaisesRegex(ValueError, 'rng'):
      sweep.verify_paired_initial_conditions(initial, 2, 2)

  def test_scores_exclude_startup_and_frozen_terminal_padding(self):
    metrics = {
        'forward_velocity': np.array([[0., .03, .04, 99.], [0., .06, .06, .06]]),
        'cycle_forward_velocity_error': np.array([[0., 0., .01, 99.], [0., 0., 0., 0.]]),
        'cycle_window_ready': np.ones((2, 4)),
        'heading_error': np.zeros((2, 4)),
    }
    for side in ('left', 'right'):
      metrics[f'desired_{side}_foot_height'] = np.full((2, 4), .01)
      metrics[f'{side}_toe_clearance'] = np.full((2, 4), .005)
      metrics[f'{side}_heel_clearance'] = np.full((2, 4), .005)
      metrics[f'{side}_foot_contact'] = np.zeros((2, 4))
    trace = {'valid': np.array([[1, 1, 1, 0], [1, 1, 1, 1]]),
             'done': np.array([[0, 0, 1, 1], [0, 0, 0, 0]]),
             'data': {'time': np.array([[1., 2., 3., 3.], [1., 2., 3., 4.]])},
             'metrics': metrics}
    rows = sweep.score(trace, [2], [30., 60.], warmup=2.)
    self.assertAlmostEqual(rows[0]['actual_mm_s'], 35.)
    self.assertAlmostEqual(rows[0]['cycle_mae_mm_s'], 5.)
    self.assertTrue(rows[0]['terminated'])
    self.assertEqual(rows[0]['duration_s'], 3.)
    self.assertEqual(rows[1]['actual_mm_s'], 60.)
    response = sweep.summarize_response(rows, [2])[0]
    self.assertIsNone(response['actual_increase_mm_s'])
    rows[0]['terminated'] = False
    response = sweep.summarize_response(rows, [2])[0]
    self.assertEqual(response['actual_increase_mm_s'], 25.)
    self.assertEqual(response['requested_increase_mm_s'], 30.)


if __name__ == '__main__':
  unittest.main()
