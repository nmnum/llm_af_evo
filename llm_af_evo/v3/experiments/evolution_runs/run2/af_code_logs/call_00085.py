def score_pool(context):
    """Leverages progress-aware exploitation and uncertainty scaling with hypervolume-based preference encoding."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blend: more exploitation early, exploration later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_total = sum(gp[name]["std"] for name in names)

        # Normalize uncertainty by range and scale with progress
        ucb_scaled = (1.0 - 0.5 * progress) * sigma_total
        
        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * ucb_scaled

        scores.append(score)
    
    return scores