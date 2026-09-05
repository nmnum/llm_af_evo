def score_pool(context):
    """Phase-aware uncertainty weighting with decay and stagnation novelty boost."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    campaign = context["campaign"]
    
    # Phase weight: decays from 1 to 0.2 as progress increases
    phase_weight = 0.8 * (1 - campaign["progress"]) + 0.2
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)

        # Apply phase-aware uncertainty weighting
        score = mu_sum + phase_weight * sigma_norm
        
        # Add stagnation-based novelty boost if campaign has been stagnant
        if campaign["stagnant_batches"] > 0:
            min_dist_to_observed = np.min(np.linalg.norm(context["X_obs"] - cand["x"], axis=1))
            score += 0.5 * (1 / (min_dist_to_observed + 1e-8))  # novelty boost
        
        scores.append(score)
    
    return scores