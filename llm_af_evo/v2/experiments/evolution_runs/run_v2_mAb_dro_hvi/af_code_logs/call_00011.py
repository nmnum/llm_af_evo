def score_pool(context):
    """Exploitation with dynamic uncertainty and novelty boost during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Dynamic weight for uncertainty: start high, decay as budget spends
    w_uncertainty = 1.0 - progress
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Combine exploitation and uncertainty with dynamic weight
        score = mu_sum + w_uncertainty * sigma_norm
        
        # Add novelty boost when stagnant to avoid local optima
        if stagnant_batches >= 3:
            dist_to_observed = np.min(np.linalg.norm(cand["x"] - context["X_obs"], axis=1))
            novel_boost = (dist_to_observed / np.sqrt(len(context["objective_names"]))) ** (-0.5)
            score += 2.0 * novel_boost
            
        scores.append(score)

    return scores