def score_pool(context):
    """Uncertainty-adjusted hypervolume signal with dynamic exploitation-exploration balance and progressive confidence scaling."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Dynamic blend: start exploiting, shift to exploring
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Progressive confidence scaling: increase uncertainty impact as progress increases
        conf_scale = 1.0 + np.tanh(2 * (progress - 0.5)) 

        adjusted_sigma = sigma_norm_sum * conf_scale

        score = w_exploit * mu_norm_sum + (1.0 - w_exploit) * adjusted_sigma 
        scores.append(score)
    
    return scores