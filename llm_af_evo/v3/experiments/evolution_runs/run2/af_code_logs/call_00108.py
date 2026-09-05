def score_pool(context):
    """Adapts exploitation and exploration via progressive uncertainty normalization combined with hypervolume signal from reference point."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Dynamic blend: start exploiting, then shift to exploring
    w_exploit = 0.2 + 0.8 * (1 - np.exp(-6 * progress))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize means and uncertainties by front range 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Progressive uncertainty scaling: increase exploration as progress increases
        unc_scale = 1.0 + (2.5 * progress) ** (1.0 - np.exp(-3 * progress))
        
        # Apply hypervolume-based signal using reference point to guide selection towards unexplored regions  
        hv_signal = sum((ref_point[i] - gp[name]["mean"]) / ref_point[i] for i, name in enumerate(names))

        score = w_exploit * mu_norm_sum + (1.0 - w_exploit) * np.sqrt(sigma_norm_sum * unc_scale) + 0.5 * hv_signal
        
        scores.append(score)
    
    return scores