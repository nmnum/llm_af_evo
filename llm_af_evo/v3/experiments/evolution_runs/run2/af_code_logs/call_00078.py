def score_pool(context):
    """Adaptive exploitation-uncertainty tradeoff with progress-aware scaling and normalized hypervolume bonus."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Blend between UCB-like exploration (early) and exploitation-focused scoring (late)
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation: sum of normalized means
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty term with adaptive scaling based on progress and objective ranges  
        sigma_scaled = sum(
            gp[name]["std"] * (1.0 + 2.0 * np.exp(-5 * progress)) / front_range[name]
            for name in names
        )

        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * sigma_scaled
        
        # Bonus based on how far the candidate's prediction is from current pareto front 
        # Normalized by range to make it scale invariant  
        dist_to_front = min(
            np.sum(np.maximum(0, ref_point[name] - gp[name]["mean"]) / front_range[name])
            for name in names
        )
        
        score += 1.5 * (np.exp(-dist_to_front) - 0.5)
            
        scores.append(score)

    return scores