def score_pool(context):
    """Progress-aware hypervolume-based exploitation with uncertainty-weighted novelty."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blend: more exploration early, exploit later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation signal based on normalized hypervolume contribution
        mu_scaled = sum(gp[name]["mean"] / front_range[name] for name in names)
        hv_contribution = 1.0 - np.prod(1.0 - (mu_scaled * w_exploit) /
                                         [ref_point[i] / front_range[names[i]] 
                                          for i, _ in enumerate(names)])
        
        # Uncertainty term with progress-dependent weighting
        sigma_weighted = sum(
            gp[name]["std"] * (2.0 + 3.0 * np.exp(-10 * progress)) /
            front_range[name]
            for name in names)

        score = hv_contribution + (1.0 - w_exploit) * sigma_weighted
        scores.append(score)
    
    return scores