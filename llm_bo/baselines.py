"""
baselines.py — Fixed-strategy and ADA-original baselines.

FixedStrategyBaseline : always uses the same strategy with fixed parameters.
ADAOriginalBaseline   : replays the actual β sequence from the 2022 coatings campaign.
"""

import numpy as np
import pandas as pd
import pathlib
from typing import Dict, Any


class FixedStrategyBaseline:
    """
    Always returns the same strategy and parameters.

    Parameters
    ----------
    strategy : one of 'ucb', 'ei', 'pi', 'thompson', 'random', 'lhs'
    params   : dict of strategy kwargs (e.g. {'beta': 0.2} for UCB)
    """

    _needs_gp_uncertainty = False  # no GP uncertainty needed

    def __init__(self, strategy: str, params: Dict[str, Any]):
        self.strategy = strategy
        self.params = params

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"strategy": self.strategy, "params": self.params}


class ADAOriginalBaseline:
    """
    Replays the actual strategy sequence used in the ADA 2022 coatings campaign.

    The `beta` column in coatings_2022.csv records which UCB β value (or 'random'
    or 'SF' for space-filling) was used at each experiment. This baseline replays
    that sequence exactly, providing a ground-truth comparison.

    Strategy mapping:
      'random' → random search
      'SF'     → LHS (space-filling)
      '2.00E-01' → UCB β=0.2
      '2.00E+01' → UCB β=20
      '4.00E+02' → UCB β=400
    """

    _needs_gp_uncertainty = False

    BETA_MAP = {
        "random": ("random", {}),
        "SF": ("lhs", {}),
        "2.00E-01": ("ucb", {"beta": 0.2}),
        "2.00E+01": ("ucb", {"beta": 20.0}),
        "4.00E+02": ("ucb", {"beta": 400.0}),
    }

    def __init__(self, beta_sequence):
        """
        Parameters
        ----------
        beta_sequence : list of (strategy_name, params) tuples in temporal order
        """
        self.beta_sequence = beta_sequence
        self._step = 0

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        idx = min(self._step, len(self.beta_sequence) - 1)
        strategy, params = self.beta_sequence[idx]
        self._step += 1
        return {"strategy": strategy, "params": params}

    @classmethod
    def from_coatings_csv(cls, csv_path: str) -> "ADAOriginalBaseline":
        """Build the baseline from the raw coatings CSV."""
        df = pd.read_csv(csv_path)
        # Deduplicate to unique experiments, preserve temporal order
        dedup = (
            df.groupby("exp_num")
            .agg({"beta": "first", "sample": "min"})
            .reset_index()
            .sort_values("sample")
            .reset_index(drop=True)
        )
        sequence = []
        for beta_str in dedup["beta"]:
            strategy, params = cls.BETA_MAP.get(str(beta_str), ("random", {}))
            sequence.append((strategy, params))
        return cls(sequence)
