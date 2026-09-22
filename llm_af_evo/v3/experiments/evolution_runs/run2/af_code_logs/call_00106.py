def score_pool(context):
    """Balances exploitation and uncertainty via progress-aware sigmoidal blending while incorporating normalized hypervolume improvement estimate."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blend: more exploitation early, exploration later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized means and uncertainties by front range 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Hypervolume improvement estimate based on candidate's distance from reference point
        hv_improvement = 1.0 - np.prod([max(0, ref_point[i] - gp[names[i]]["mean"]) 
                                         for i in range(len(names))])

        score = w_exploit * mu_norm_sum + (1.0 - w_exploit) * sigma_norm_sum
        # Adjust by hypervolume signal to favor candidates that expand dominated space more effectively  
        scores.append(score * hv_improvement)
    
    return scores