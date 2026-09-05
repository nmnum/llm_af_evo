def score_pool(context):
    """Progressively balances exploitatation and uncertainty via dynamic weighting with hypervolume-aware normalization."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Dynamic exploitation weight that shifts from UCB-like to pure exploration
    w_exploit = 0.2 + 0.8 * (1 - np.exp(-4 * progress))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize means and uncertainties by front range 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Scale uncertainty dynamically with progress, but apply sqrt to reduce over-weighting
        scaled_sigma = np.sqrt(sigma_norm_sum * (1.0 + 3.0 * progress))
        
        score = w_exploit * mu_norm_sum + (1 - w_exploit) * scaled_sigma
        
        scores.append(score)
    
    return scores