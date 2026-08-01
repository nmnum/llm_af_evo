def score_pool(context):
    """Exploitation with dynamic weighting and novelty penalty to balance progress and diversity."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    
    # Dynamic weight for exploitation vs uncertainty
    w_exploit = 1.0 - 0.5 * progress  # Start with more exploitation, reduce as we progress
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Combine exploitation and uncertainty with dynamic weight
        score = w_exploit * mu_sum + (1.0 - w_exploit) * sigma_norm
        
        # Add novelty penalty: candidates closer to observed points get lower scores
        if X_obs.size > 0:
            dist_to_observed = np.min(np.linalg.norm(cand["x"] - X_obs, axis=1))
            novelty_penalty = 1.0 / (1.0 + dist_to_observed)
            score *= novelty_penalty
        
        scores.append(score)
    
    return scores