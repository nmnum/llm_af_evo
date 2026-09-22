def score_pool(context):
    """Balances exploitation and exploration via adaptive uncertainty scaling modulated by progress-dependent hypervolume signal."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Progress-aware blending: early exploit, late explore
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Normalize uncertainty by current progress-dependent scaling
        adaptive_sigma = np.sqrt(sigma_norm_sum * (1.0 + 2.0 * progress))

        score = w_exploit * mu_sum + (1.0 - w_exploit) * adaptive_sigma
        
        scores.append(score)
    
    return scores