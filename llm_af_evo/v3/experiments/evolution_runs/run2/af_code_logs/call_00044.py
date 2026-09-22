def score_pool(context):
    """Progressively shifts between hypervolume-based acquisition and uncertainty-weighted exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Use a smooth sigmoidal transition from UCB-like early exploration to hypervolume-based later exploitation
    w_exploit = 1.0 / (1.0 + np.exp(-24 * (progress - 0.5)))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized mean sum for exploitation signal  
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty term (normalized standard deviation) 
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * sigma_norm
        scores.append(score)
    
    return scores