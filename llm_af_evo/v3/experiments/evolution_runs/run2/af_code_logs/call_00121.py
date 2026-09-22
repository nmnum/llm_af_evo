def score_pool(context):
    """Exploits predicted means with uncertainty-aware hypervolume signal and dynamic exploitation weight based on progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Start with strong exploration, taper off towards later stages
    w_exploit = 0.2 + (1 - np.exp(-5 * progress)) * 0.8

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Apply a dynamic scaling factor to uncertainty based on progress and candidate quality
        scaled_sigma = np.sqrt(sigma_norm_sum * (1.0 + 3.0 * w_exploit))

        score = w_exploit * mu_sum - (1 - w_exploit) * scaled_sigma
        
        scores.append(score)
    
    return scores