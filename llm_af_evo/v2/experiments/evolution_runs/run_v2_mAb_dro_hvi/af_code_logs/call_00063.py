def score_pool(context):
    """Exploitation with uncertainty-aware ranking blended with novelty reward: sum of means minus scaled std penalty, plus distance from latest observed points."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Progress-driven exploration weight (exploit more as we go)
    progress = context["campaign"]["progress"] 
    exploit_weight = 1.0 - max(0., (progress * 2.0) - 1.) ** 2
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation term
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        # Uncertainty penalty (scaled by progress-driven weight)  
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Novelty bonus: distance from nearest previously observed point 
        cand_x = cand["x"]
        if len(context["X_obs"]) > 0:
            distances = np.linalg.norm(context["X_obs"] - cand_x, axis=1)  
            novelty_bonus = np.min(distances)
        else:
            novelty_bonus = 1.0
            
        # Combine all terms
        score = mu_sum - exploit_weight * sigma_norm + novelty_bonus
        
        scores.append(score)

    return scores