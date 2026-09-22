def score_pool(context):
    """Progress-aware exploitation with uncertainty-driven hypervolume potential and dynamic blending."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Blend ratio: more exploit as we near the end, but use a steeper curve
    w_exploit = 1.0 / (1.0 + np.exp(-8 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized means 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty term with dynamic scaling and progress-aware penalty
        sigma_scaled = sum(
            gp[name]["std"] * (1.0 + 3.0 * np.exp(-4 * progress)) / front_range[name]
            for name in names
        )

        score = w_exploit * mu_norm_sum - (1.0 - w_exploit) * sigma_scaled 
        scores.append(score)
    
    return scores