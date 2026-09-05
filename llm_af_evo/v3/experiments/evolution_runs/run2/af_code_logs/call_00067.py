def score_pool(context):
    """Adapts uncertainty sensitivity based on progress and candidate proximity to the current Pareto front."""
    names = context["objective_names"]
    pareto_front = context["pareto_front"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Scale exploration weight with a sigmoid function of campaign progress
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation: normalized mean sum 
        mu_sum_norm = sum(gp[name]["mean"] / (ref_point[i] - pareto_front.min(axis=0)[i]) 
                          for i, name in enumerate(names))
                
        # Uncertainty-aware exploitation with dynamic scaling
        sigma_scaled = 1.0 + np.sum([gp[name]["std"]/np.sqrt(2) * w_exploit  
                                     for name in names])
        
        score = mu_sum_norm / (sigma_scaled)
        scores.append(score)

    return scores