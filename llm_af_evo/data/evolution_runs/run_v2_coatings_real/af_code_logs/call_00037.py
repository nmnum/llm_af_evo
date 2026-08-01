def score_pool(context):
    """Exploitation with progress-aware uncertainty bonus: UCB-style score with weight decaying as training progresses, and novelty bonus to avoid stagnation."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Decay uncertainty weight as progress increases, favouring exploitation later
        ucb_weight = 2.0 * (1.0 - progress)
        # Add a novelty bonus if we've been stagnant recently
        novelty_bonus = 0.0
        if stagnant_batches > 0:
            min_dist = np.min(np.linalg.norm(context["X_obs"] - cand["x"], axis=1))
            novelty_bonus = 1.0 / (1.0 + min_dist)
        scores.append(mu_sum + ucb_weight * sigma_sum + novelty_bonus)
    return scores