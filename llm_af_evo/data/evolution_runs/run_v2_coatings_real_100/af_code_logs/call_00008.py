def score_pool(context):
    """Blend UCB-style exploration with dynamic uncertainty weighting and novelty boost during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    X_obs = context["X_obs"]
    campaign = context["campaign"]
    
    progress = campaign["progress"]
    stagnant_batches = campaign["stagnant_batches"]
    
    # Dynamic uncertainty weight: decay as progress increases
    uncertainty_weight = 1.0 - progress * 0.5
    
    # Increase uncertainty weight during stagnation
    if stagnant_batches > 0:
        uncertainty_weight *= 1.0 + min(stagnant_batches, 3) * 0.2
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # UCB-style score
        ucb_score = mu_sum + uncertainty_weight * sigma_norm
        
        # Add novelty bonus if not too early
        if len(X_obs) > 0 and progress > 0.2:
            cand_x = cand["x"]
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            novelty = np.min(distances)
            ucb_score += 0.3 * novelty
        
        scores.append(ucb_score)
    
    return scores