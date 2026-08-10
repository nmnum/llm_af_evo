def score_pool(context):
    """Weight uncertainty heavily early, decay as budget spends, with novelty boost during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Phase-aware weight for uncertainty term
    w_uncertainty = 1.0 - progress
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Apply phase-aware uncertainty weight
        score = mu_sum + w_uncertainty * sigma_norm
        
        # Add novelty boost if stagnant
        if stagnant_batches >= 3:
            dist_to_observed = np.min(np.linalg.norm(cand["x"] - context["X_obs"], axis=1))
            novel_penalty = (dist_to_observed / np.sqrt(len(context["objective_names"]))) ** 2
            score -= 0.5 * novel_penalty
            
        scores.append(score)
    
    return scores