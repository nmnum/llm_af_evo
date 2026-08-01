def score_pool(context):
    """Exploitation with uncertainty-weighted novelty: blend predicted means and normalized uncertainty, adjusted by campaign progress, and penalize candidates close to already observed points."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Early progress: balance exploitation and exploration; later: exploit more
        w_exploit = 0.5 + 0.5 * (1 - progress)
        score = w_exploit * mu_sum + (1 - w_exploit) * sigma_norm
        
        # Add novelty penalty: candidates close to observed points get lower scores
        if len(X_obs) > 0:
            dist_to_observed = np.min(np.linalg.norm(cand["x"] - X_obs, axis=1))
            # Normalize the distance by the range of features to make it scale-invariant
            feature_range = np.max(X_obs, axis=0) - np.min(X_obs, axis=0)
            if np.any(feature_range == 0):
                feature_range[feature_range == 0] = 1  # Avoid division by zero
            normalized_dist = dist_to_observed / np.mean(feature_range)
            # Inverse relationship: closer points get lower scores
            novelty_penalty = 1.0 / (1.0 + normalized_dist)
            score *= novelty_penalty
            
        scores.append(score)
    return scores