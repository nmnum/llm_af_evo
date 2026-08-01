def score_pool(context):
    """Exploitation with progress-aware uncertainty bonus: UCB-style score with weight decaying as training progresses, plus novelty reward."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Decay uncertainty weight as progress increases, favouring exploitation later
        ucb_weight = 2.0 * (1.0 - progress)
        # Add novelty bonus: candidates farther from observed points get higher scores
        min_dist = np.min(np.linalg.norm(cand["x"] - context["X_obs"], axis=1)) if len(context["X_obs"]) > 0 else 0.0
        novelty_bonus = 0.1 * (1.0 / (min_dist + 1e-8))  # Avoid division by zero
        scores.append(mu_sum + ucb_weight * sigma_sum + novelty_bonus)
    return scores