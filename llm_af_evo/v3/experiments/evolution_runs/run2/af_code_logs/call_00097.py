def score_pool(context):
    """Adapts exploration preference based on recent progress stagnation and combines mean prediction with uncertainty scaled by front diversity."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    campaign = context["campaign"]
    
    # Progress-driven exploitation weight: start strong, taper off
    w_exploit = 0.3 + 0.7 * (1 - np.exp(-5 * campaign["progress"]))
    
    # Increase exploration if progress has stalled recently  
    stagnation_factor = min(2.0, 1.0 + campaign["stagnant_batches"] * 0.2)
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names) 
        sigma_sum = sum(gp[name]["std"] for name in names)

        # Scale uncertainty by how much the front varies per objective and stagnation
        diversity_factor = np.mean([front_range[name] for name in names])
        scaled_sigma = sigma_sum / (diversity_factor + 1e-8) * stagnation_factor
        
        score = w_exploit * mu_sum + (1 - w_exploit) * scaled_sigma 
        scores.append(score)
    
    return scores