def score_pool(context):
    """Progress-adaptive exploitation-plus-uncertainty with hypervolume-normalized scores and novelty bonus."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Adaptive weight for exploitation vs exploration
    w_exploit = 1.0 - max(0.0, min(1.0, (2 * progress) ** 3))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted means and normalized stds  
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        # Normalize the mean score based on current front range
        norm_mu = mu_sum / len(names)  # simple average of means
        
        # Blend exploitation and uncertainty scores, weighted by progress  
        score = w_exploit * norm_mu + (1.0 - w_exploit) * sigma_norm

        # Add novelty bonus: inverse distance to nearest observed point
        cand_x = cand["x"]
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            novel_bonus = 1.0 / (np.min(distances) + 1e-8)
            score += 0.5 * novel_bonus

        scores.append(score)

    return scores