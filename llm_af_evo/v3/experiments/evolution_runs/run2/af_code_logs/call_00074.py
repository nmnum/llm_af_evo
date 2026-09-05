def score_pool(context):
    """Leverages progressive utility scaling and uncertainty-aware hypervolume signal for adaptive exploration-exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Progress-driven exploitation weight: start with strong UCB-like behavior, taper off
    w_exploit = 0.3 + 0.7 * (1 - np.exp(-5 * progress))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize means and uncertainties by front range 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Apply dynamic scaling to uncertainty based on progress
        scaled_sigma = np.sqrt(sigma_norm_sum * (1.0 + 2.5 * progress))
        
        score = w_exploit * mu_norm_sum + (1 - w_exploit) * scaled_sigma
        
        scores.append(score)
    
    return scores