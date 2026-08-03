"""
controllers/rule_router.py — Threshold-based strategy router.

Implements the same routing logic as approach_d but using fixed thresholds
instead of an LLM. This is the key comparator for the LLM router: it receives
exactly the same campaign-state signals and applies deterministic rules.

If rule_router ≈ approach_d everywhere: the LLM adds no routing value.
If approach_d > rule_router on ambiguous landscapes: LLM reasoning helps.
If rule_router > fixed baselines: routing itself has value, regardless of LLM.

Thresholds (initial guesses, to be calibrated on Phase 1 data):
  lengthscale_norm < 0.15  → landscape too rough for GP, use LHS
  lengthscale_norm > 0.30  → landscape structured, use EGBO
  best_normalised  > 0.90  → near-optimal, switch to exploitation
  n_obs / n_dims   < 3     → too few obs for reliable GP, use LHS
  budget / n_dims  < 5     → overall budget too small for GP, use LHS
  improvement_rate < 0.05  → stagnating, try EGBO for diversity
"""

from typing import Any, Dict


class RuleRouterController:
    """
    Threshold-based strategy router. No LLM calls.

    Uses the same signals as ApproachDController:
      lengthscale_norm, best_normalised, improvement_rate,
      n_obs, n_dims, budget.

    Parameters
    ----------
    ls_rough   : lengthscale_norm below this → LHS (landscape too flat/rough)
    ls_smooth  : lengthscale_norm above this → EGBO (landscape structured)
    best_exploit: best_normalised above this → exploit (near-optimal)
    min_obs_per_dim: n_obs/n_dims below this → LHS (GP unreliable)
    min_budget_per_dim: budget/n_dims below this → LHS (insufficient budget for GP)
    stagnation_rate: improvement_rate below this → EGBO (escape stagnation)
    """

    _needs_gp_uncertainty = True  # needs lengthscale_norm

    def __init__(
        self,
        ls_rough:            float = 0.05,   # lowered: pareto landscapes genuinely rough
        ls_smooth:           float = 0.25,   # lowered: easier to detect structure
        best_exploit:        float = 0.90,
        min_obs_per_dim:     float = 3.0,
        min_budget_per_dim:  float = 5.0,
        stagnation_rate:     float = 0.05,
    ):
        self.ls_rough           = ls_rough
        self.ls_smooth          = ls_smooth
        self.best_exploit       = best_exploit
        self.min_obs_per_dim    = min_obs_per_dim
        self.min_budget_per_dim = min_budget_per_dim
        self.stagnation_rate    = stagnation_rate

        self._stagnation_count = 0  # consecutive stagnating calls

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ls          = context.get("lengthscale_norm", 0.5)
        best_norm   = context["best_normalised"]
        improv_rate = context["improvement_rate"]
        n_obs       = context["n_obs"]
        n_dims      = context.get("n_dims", context["bounds"].shape[0])
        budget      = context["budget"]

        obs_per_dim    = n_obs / max(n_dims, 1)
        budget_per_dim = budget / max(n_dims, 1)

        # Track consecutive stagnation
        if improv_rate < self.stagnation_rate:
            self._stagnation_count += 1
        else:
            self._stagnation_count = 0

        # Rule 1: already near-optimal → exploit
        if best_norm > self.best_exploit:
            return {"strategy": "ucb", "params": {"beta": 0.1}}

        # Rule 2: too few observations for reliable GP → space-fill
        if obs_per_dim < self.min_obs_per_dim:
            return {"strategy": "lhs", "params": {}}

        # Rule 3: budget too small for GP to pay off → space-fill
        if budget_per_dim < self.min_budget_per_dim:
            return {"strategy": "lhs", "params": {}}

        # Rule 4: landscape too rough for GP → space-fill
        if ls < self.ls_rough:
            return {"strategy": "lhs", "params": {}}

        # Rule 5: stagnating for 2+ consecutive calls → EGBO for diversity escape
        if self._stagnation_count >= 2:
            return {"strategy": "egbo", "params": {}}

        # Rule 6: landscape is structured → EGBO
        if ls > self.ls_smooth and budget_per_dim > self.min_budget_per_dim:
            return {"strategy": "egbo", "params": {}}

        # Default for ambiguous landscapes (ls between rough and smooth thresholds):
        # Use EGBO if obs_per_dim is sufficient (GP calibrated enough)
        # Use UCB otherwise
        if obs_per_dim >= self.min_obs_per_dim * 2:
            return {"strategy": "egbo", "params": {}}
        # UCB with exploration decaying over progress
        progress = context["step"] / context["budget"]
        gp_unc = context.get("gp_uncertainty", 0.3)
        beta = max(0.5, min(50.0, 10.0 * gp_unc * (1.0 - progress)))
        return {"strategy": "ucb", "params": {"beta": round(beta, 2)}}
