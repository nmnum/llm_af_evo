def score_pool(context):
    """Progress-adaptive hypervolume estimate combined with inverse distance novelty."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]
    
    # Use a simple sigmoid to shift weight from uncertainty to exploitation
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Estimated hypervolume contribution based on GP mean and reference point
        hv_contrib = np.prod(np.maximum(0, ref_point - [gp[name]["mean"] for name in names]))
        
        # Normalized uncertainty (std scaled by front range)
        sigma_norm_sum = sum(gp[name]["std"]/front_range[name] for name in names)

        # Combine exploitation and novelty with progress-adaptive weights
        score = w_exploit * hv_contrib + (1.0 - w_exploit) * (-sigma_norm_sum)  # negative because lower uncertainty is better

        scores.append(score)
    
    return scores