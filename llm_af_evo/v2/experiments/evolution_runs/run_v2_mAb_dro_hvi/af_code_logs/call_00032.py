def score_pool(context):
    """Exploitation with novelty bonus: rank by predicted objective sum, then boost candidates far from existing observations."""
    names = context["objective_names"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        # Compute distance to nearest observed point
        x_cand = cand["x"]
        if len(context["X_obs"]) == 0:
            novelty_bonus = 1.0
        else:
            distances = np.linalg.norm(context["X_obs"] - x_cand, axis=1)
            min_distance = np.min(distances)
            # Normalize by the range of features to make this scale-invariant
            feature_range = np.max(context["X_obs"], axis=0) - np.min(context["X_obs"], axis=0)
            if not np.any(feature_range == 0):
                normalized_dist = min_distance / np.mean(feature_range)
            else:
                # If all features are the same, set distance to max possible
                normalized_dist = 1.0
                
            novelty_bonus = 1.0 + (1.0 - normalized_dist) * 2.0
        
        scores.append(mu_sum * novelty_bonus)
    return scores