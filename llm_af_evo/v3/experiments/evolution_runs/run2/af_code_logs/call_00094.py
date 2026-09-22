def score_pool(context):
    """Progress-aware blend of normalized mean and scaled uncertainty with dynamic exponentiation for hypervolume signal."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blending: more exploitation early, exploration later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized means and uncertainties by front range 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Dynamic exponentiation on uncertainty to increase sensitivity at later stages
        dynamic_exp = 1.0 + (2.0 * progress) 
        score = w_exploit * mu_norm_sum + (1.0 - w_exploit) * (sigma_norm_sum ** dynamic_exp)
        
        scores.append(score)
    
    return scores