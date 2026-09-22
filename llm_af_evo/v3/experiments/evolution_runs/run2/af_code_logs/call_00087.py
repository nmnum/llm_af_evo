def score_pool(context):
    """Progress-aware blend of normalized mean and uncertainty with dynamic exponentiation for adaptive exploration-exploitation balance."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]

    # Dynamic blending: start exploit, shift to explore
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Use exponential scaling on uncertainty to increase its impact as progress increases
        exp_factor = np.exp(2 * progress)  
        adjusted_sigma = (1.0 + 3.0 * exp_factor) * sigma_norm_sum

        score = w_exploit * mu_norm_sum + (1.0 - w_exploit) *adjusted_sigma 
        scores.append(score)
    
    return scores