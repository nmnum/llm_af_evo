def score_pool(context):
    """Adaptive hypervolume-based uncertainty scaling with progressive exploitation balance."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Dynamic exploit-weight based on campaign progress
    w_exploit = 0.2 + 0.8 * (1 - np.exp(-6 * progress))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize means by front range 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty scaled with exponential decay based on progress
        sigma_scaled = sum(
            gp[name]["std"] * np.exp(-3.0 * progress) / front_range[name]
            for name in names
        )

        score = w_exploit * mu_norm_sum + (1 - w_exploit) * sigma_scaled
        scores.append(score)
    
    return scores