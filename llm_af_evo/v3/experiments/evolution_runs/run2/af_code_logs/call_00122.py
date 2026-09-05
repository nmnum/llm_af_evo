def score_pool(context):
    """Progress-aware exploitation with uncertainty-weighted hypervolume gain estimation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Blend exploit vs explore based on campaign progress
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation: normalized mean objectives 
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty-aware hypervolume gain estimate
        sigma_prod = np.prod([1.0 + 2.0 * gp[name]["std"] / front_range[name]
                              for name in names])
        
        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * np.log(sigma_prod)
        
        scores.append(score)

    return scores