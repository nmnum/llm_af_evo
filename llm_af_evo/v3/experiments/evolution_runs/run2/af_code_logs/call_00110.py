def score_pool(context):
    """Adaptive hypervolume gain estimator with progressive uncertainty modulation and dynamic exploitation balance."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Dynamic blend: start biased towards exploration, shift to exploitation
    w_exploit = 0.3 + 0.7 * np.tanh(4 * (progress - 0.5))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted objective means and uncertainties 
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Progressive uncertainty modulation: reduce influence of std as progress increases
        modulated_sigma = np.sqrt(sigma_norm_sum) * (1.0 - 0.5 * progress)
        
        score = w_exploit * mu_sum + (1.0 - w_exploit) * modulated_sigma
        
        scores.append(score)

    return scores