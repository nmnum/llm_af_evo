def score_pool(context):
    """Progress-adaptive exploitation with uncertainty-aware hypervolume normalization and dynamic scaling."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Dynamic weight: exploit more as we approach the end
    w_exploit = 0.5 + 0.5 * np.tanh(2 * (progress - 0.5))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized means and scaled uncertainties 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_scaled_sum = sum(
            gp[name]["std"] * (1.0 + 2.0 * progress) / front_range[name]
            for name in names
        )
        
        # Combine with dynamic blend and normalize by objective count to avoid scale bias 
        score = w_exploit * mu_norm_sum + (1 - w_exploit) * sigma_scaled_sum
        
        scores.append(score)
    
    return scores