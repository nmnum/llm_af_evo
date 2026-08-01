def score_pool(context):
    """Blend UCB-style exploitation with novelty reward, scaling uncertainty weight by progress and boosting selection during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    X_obs = context["X_obs"]
    campaign = context["campaign"]
    progress = campaign["progress"]
    stagnant_batches = campaign["stagnant_batches"]
    
    # Dynamic uncertainty weight that decreases with progress
    ucb_weight = 2.0 * (1.0 - progress)
    
    # Boost selection if stagnating
    stagnation_boost = 1.0 if stagnant_batches >= 3 else 0.0
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        ucb_score = mu_sum + ucb_weight * sigma_norm
        
        # Novelty reward: inverse distance to nearest observed point
        cand_x = cand["x"]
        if X_obs.size == 0:
            novelty = 1.0
        else:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            min_distance = np.min(distances)
            novelty = 1.0 / (1e-8 + min_distance)
        
        # Final score combines UCB, novelty, and stagnation boost
        score = ucb_score + 0.5 * novelty + stagnation_boost
        scores.append(score)
    
    return scores