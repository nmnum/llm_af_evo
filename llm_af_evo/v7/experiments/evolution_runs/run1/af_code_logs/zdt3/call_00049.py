def modifier(context):
    """Combines uncertainty bonus with front-distance-based novelty incentive, scaled by campaign progress."""
    import numpy as np
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    
    # Uncertainty component (like Parent A)
    weight_uncert = 0.3993 * (1.0 - progress)
    uncert_terms = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_sum = sum(gp[name]["std"] / front_range[name] for name in names)
        uncert_terms.append(weight_uncert * sigma_sum)

    # Novelty component (like Parent B, but simplified and combined)
    k = 3
    bonus_factor_novelty = 0.2815 
    novelty_terms = []
    
    pf = context["pareto_front"]
    if len(pf) < k:
        use_points = np.vstack([context["Y_obs"], pf])
        unique_indices, _ = np.unique(use_points, axis=0, return_index=True)
        use_points = use_points[unique_indices]
    else:
        use_points = pf
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        if len(use_points) == 0: 
            mean_dist = 1.0
        else:
            diffs = use_points - gp_mean[None, :]
            dists_sq = np.sum(diffs**2, axis=1)
            sorted_indices = np.argsort(dists_sq)[:min(k, len(use_points))]
            if not sorted_indices.size:
                mean_dist = 1.0  
            else: 
                distances = list(np.sqrt(dists_sq[sorted_indices]))
                mean_dist = np.mean(distances) if distances and len(distances)>0 else 1.0

        novelty_terms.append(min(bonus_factor_novelty * mean_dist, 0.6049))

    # Combine both terms
    return [u + n for u, n in zip(uncert_terms, novelty_terms)]