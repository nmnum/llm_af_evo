def score_pool(context):
    """Balances exploitation via normalized mean and exploration via uncertainty scaled by reference point distance with progress-adaptive weighting."""
    names = context["objective_names"]
    ref_point = context["ref_point"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blend: more exploitation early, exploration later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize means by front range 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        
        # Compute uncertainty scaled by distance to reference point
        sigma_ref_scaled_sum = 0.0
        for name in names:
            dist_to_ref = ref_point[names.index(name)] - gp[name]["mean"]
            scaling_factor = max(1e-6, dist_to_ref / front_range[name])
            sigma_ref_scaled_sum += gp[name]["std"] * scaling_factor

        # Combine with dynamic blend
        score = w_exploit * mu_norm_sum + (1.0 - w_exploit) * np.sqrt(sigma_ref_scaled_sum)
        
        scores.append(score)

    return scores