"""Regression tests for the year-end hard SOC terminal in feedback DP."""

import unittest

import numpy as np

from solve_problem3 import (
    FeedbackNoActionError,
    RunConfig,
    StorageParams,
    execute_feedback_block,
    feedback_value_function,
    make_soc_grid,
)


class YearEndFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.params = StorageParams()
        self.config = RunConfig(
            soc_step=100.0,
            feedback_policy="grid-boundary",
            forbid_emergency_charging=True,
        )

    def test_forecast_surplus_does_not_make_physical_terminal_unreachable(self):
        periods = 24
        scenario_net = np.zeros((2, periods))
        scenario_net[0, :12] = 450.0
        grid = make_soc_grid(self.params, self.config.soc_step, [6000.0])
        value, _, _ = feedback_value_function(
            np.zeros(periods),
            scenario_net,
            np.array([0.9, 0.1]),
            np.ones(periods),
            grid,
            self.params,
            self.config,
            6000.0,
            0.0,
            True,
        )
        self.assertTrue(np.isfinite(value[0, np.searchsorted(grid, 10500.0)]))

    def test_actual_dispatch_still_obeys_dominance_and_terminal(self):
        periods = 24
        actual = np.zeros(periods)
        actual[:12] = 450.0
        scenarios = np.stack([actual, np.zeros(periods)])
        result = execute_feedback_block(
            committed_g=np.zeros(periods),
            actual_net=actual,
            scenario_net=scenarios,
            probabilities=np.array([0.9, 0.1]),
            price=np.ones(periods),
            soc_start=10500.0,
            terminal_target=6000.0,
            terminal_soc_value=0.0,
            hard_terminal=True,
            params=self.params,
            config=self.config,
        )
        self.assertAlmostEqual(result["soc_end"][-1], 6000.0, places=6)
        self.assertFalse(np.any((result["unused"] > 1e-7) & (result["discharge"] > 1e-7)))
        self.assertFalse(np.any((result["emergency"] > 1e-7) & (result["charge"] > 1e-7)))

    def test_genuinely_unreachable_actual_terminal_still_fails(self):
        periods = 2
        with self.assertRaises(FeedbackNoActionError):
            execute_feedback_block(
                committed_g=np.zeros(periods),
                actual_net=np.zeros(periods),
                scenario_net=np.zeros((1, periods)),
                probabilities=np.ones(1),
                price=np.ones(periods),
                soc_start=10500.0,
                terminal_target=6000.0,
                terminal_soc_value=0.0,
                hard_terminal=True,
                params=self.params,
                config=self.config,
            )


if __name__ == "__main__":
    unittest.main()
