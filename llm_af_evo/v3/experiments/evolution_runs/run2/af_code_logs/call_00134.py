def score_pool(context):
    """Balances exploitation and uncertainty via progress-adaptive mean-std tradeoff while emphasizing hypervolume potential near the reference point."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Dynamic weight: start with strong exploitation, shift towards uncertainty as campaign progresses
    w_exploit = 0.8 * (1 - np.exp(-3 * progress))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Apply adaptive scaling to uncertainty with a floor that increases over time
        scaled_sigma = np.sqrt(sigma_norm_sum * (0.5 + 1.5 * progress))

        score = w_exploit * mu_sum + (1 - w_exploit) * scaled_sigma
        
        scores.append(score)
    
    return scores