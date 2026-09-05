def score_pool(context):
    """Combines progress-aware exploitation with uncertainty-weighted hypervolume difference to prioritize candidates near the pareto front's edge."""
    names = context["objective_names"]
    ref_point = context["ref_point"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]

    # Blend: start exploit, shift towards explore
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_scaled_sum = sum(
            gp[name]["std"] * (1 + 2.0*progress) / front_range[name]
            for name in names
        )

        # Predicted hypervolume difference scaled by uncertainty 
        pred_objs = np.array([gp[name]["mean"] for name in names])
        hv_diff = max(0, ref_point[0] - pred_objs[0]) * max(0, ref_point[1] - pred_objs[1])

        score = w_exploit * mu_norm_sum + (1.0 - w_exploit) * sigma_scaled_sum
        
        # Add a term that rewards candidates with higher predicted HV difference
        hv_term = 2.0 * hv_diff / np.prod(ref_point)
        
        scores.append(score + hv_term)

    return scores