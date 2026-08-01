def score_pool(context):
    """Blend UCB-style exploration with dynamic uncertainty weighting and novelty bonus during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    X_obs = context["X_obs"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Dynamic uncertainty weight: decrease as progress increases
    uncertainty_weight = 2.0 * (1.0 - progress)
    
    # Increase uncertainty weight during stagnation
    if stagnant_batches > 0:
        uncertainty_weight *= 1.0 + 0.3 * min(stagnant_batches, 4)
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # UCB-style score
        ucb_score = mu_sum + uncertainty_weight * sigma_norm
        
        # Novelty bonus: distance to nearest observed point
        cand_x = cand["x"]
        if len(X_obs) == 0:
            novelty = 0.0
        else:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            novelty = np.min(distances)
        
        # Apply novelty boost only when stagnating
        novelty_bonus = 0.0
        if stagnant_batches > 0:
            novelty_bonus = 0.5 * novelty
        
        scores.append(ucb_score + novelty_bonus)
    
    return scores