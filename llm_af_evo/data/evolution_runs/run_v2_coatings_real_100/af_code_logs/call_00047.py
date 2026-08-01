def score_pool(context):
    """Exploitation with adaptive uncertainty bonus: sum of means plus UCB-style uncertainty scaled by progress, with novelty penalty."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adaptive UCB weight: higher early, decays to 0.5 near end
        ucb_weight = 2.0 * (1.0 - progress) + 0.5
        score = mu_sum + ucb_weight * sigma_sum
        # Add a novelty penalty based on distance to nearest observed point
        if len(context["X_obs"]) > 0:
            distances = np.linalg.norm(context["X_obs"] - cand["x"], axis=1)
            min_distance = np.min(distances)
            # Normalize by the range of observed features
            feature_range = np.max(context["X_obs"], axis=0) - np.min(context["X_obs"], axis=0)
            normalized_distance = min_distance / (np.mean(feature_range) + 1e-8)
            score -= 0.1 * normalized_distance
        scores.append(score)
    return scores