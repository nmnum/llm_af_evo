def score_pool(context):
    """Adaptive exploitation-uncertainty balance with dynamic reference point scaling and normalized hypervolume gain."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Dynamically adjust the reference point based on campaign progress to shift focus
    adjusted_ref = np.array([
        ref_point[i] + (0.5 - 0.3 * progress) * front_range[name]
        for i, name in enumerate(names)
    ])

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize candidate means and stds by the current front range
        mu_norm = np.array([gp[name]["mean"] / front_range[name] for name in names])
        sigma_norm = np.array([gp[name]["std"] / front_range[name] for name in names])

        # Use adjusted reference point to compute hypervolume contribution estimate 
        cand_ref_dists = ref_point - mu_norm
        hv_contribution = np.prod(np.maximum(cand_ref_dists, 0.0))

        # Combine exploitation and uncertainty with a dynamic balance factor  
        w_exploit = 1.0 / (1.0 + np.exp(-8 * (progress - 0.4)))
        
        score = (
            w_exploit * hv_contribution +
            (1 - w_exploit) * np.sum(sigma_norm)
        )
            
        scores.append(score)

    return scores