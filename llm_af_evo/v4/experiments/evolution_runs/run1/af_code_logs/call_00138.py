def score_pool(context):
    """Blend acquisition value with inverse uncertainty and progress-aware exploitation."""
    
    names = context["objective_names"]
    scores = []
    
    # Base score from acq_value_norm (already hypervolume improvement estimate)
    base_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Add a bonus based on inverse uncertainty, scaled by progress
    inv_uncertainty_bonus = [
        sum(1.0 / (gp_posterior[name]["std"] + 1e-8) 
            for name in names)
        for gp_posterior in [cand["gp_posterior"] for cand in context["pool"]]
    ]
    
    # Progress-aware exploitation: increase the bonus weight as campaign progresses
    progress = context["campaign"]["progress"]
    exploit_weight = max(0.1, 1.0 - progress)  # Start low (explore), decrease over time
    
    final_scores = [
        base + exploit_weight * inv_uncertainty_bonus[i]
        for i, (base, inv_unce in zip(base_scores, inv_uncertainty_bonus))
    ]
    
    return list(final_scores)