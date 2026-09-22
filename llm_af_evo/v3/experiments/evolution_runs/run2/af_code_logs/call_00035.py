def score_pool(context):
    """Progressively shifts between hypervolume-based acquisition and uncertainty-weighted exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Use a sigmoidal blend to shift from pure UCB-like exploration early on, towards hypervolume estimation later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation signal: mean of objectives, normalized by front range
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty term (normalized standard deviation) 
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * sigma_norm
        scores.append(score)
    
    return scores