def score_pool(context):
    """Balances exploitation and uncertainty using dynamic weighting derived from Pareto front geometry and progress to encourage diverse yet high-value candidates."""
    names = context["objective_names"]
    pareto_front = context["pareto_front"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Compute hypervolume of current front relative to reference point
    if len(pareto_front) >= 2:
        hv_current = np.prod(ref_point - np.min(pareto_front, axis=0))
    else:
        hv_current = 1.0

    # Progress-aware blend: more exploitation as we near the end and when front is large (more stable)
    w_exploit = progress * (len(pareto_front) / max(1., len(context["X_obs"]) - 2))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized mean and uncertainty
        mu_norm_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] for name in names)

        score = w_exploit * mu_norm_sum + (1.0 - w_exploit) * np.sqrt(sigma_norm_sum)
        
        scores.append(score)
    
    return scores