def score_pool(context):
    """Balances exploitation and uncertainty via progress-aware blending while incorporating hypervolume signal from predicted candidates."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Blend between UCB-like exploration (early) and exploitation-focused scoring (late)
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation: sum of normalized means
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty: total std scaled by range  
        sigma_total = sum(gp[name]["std"] for name in names)

        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * sigma_total
        scores.append(score)
    
    return scores