def score_pool(context):
    """Combines uncertainty-weighted mean with progress-adaptive hypervolume difference to favor candidates expanding dominated regions."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Use a sigmoidal blend between pure exploitation and uncertainty-based exploration
    w_exploit = 0.2 + 0.8 * (1 - np.exp(-5 * progress))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Estimate hypervolume contribution using reference point and normalized uncertainties
        hv_diff = 1.0 - np.prod([max(0, ref_point[i] - gp[names[i]]["mean"]) 
                                * (1 + sigma_norm_sum)  
                                / max(ref_point[i], front_range[name]) for i, name in enumerate(names)])

        score = w_exploit * mu_sum + (1 - w_exploit) * hv_diff
        
        scores.append(score)
    
    return scores