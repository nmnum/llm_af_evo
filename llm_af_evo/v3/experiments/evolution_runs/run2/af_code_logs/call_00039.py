def score_pool(context):
    """Balances uncertainty-weighted exploitation and hypervolume-based acquisition with progress-aware blending."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Blend between UCB-like exploration (early) and HV improvement (late)
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized mean objective values
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Normalized uncertainty (standard deviation) 
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * sigma_norm
        scores.append(score)
    
    return scores