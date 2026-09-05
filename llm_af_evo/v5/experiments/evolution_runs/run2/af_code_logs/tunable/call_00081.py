def score_pool(context):
    """Blend acquisition value with a front-density-weighted uncertainty signal that emphasizes expanding underexplored regions of objective space."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    
    # Compute candidate proximity to Pareto front (inverse density)
    if len(context["pareto_front"]) < 2:
        # Fallback for early campaign: use all observations
        ref_points = context["Y_obs"] 
    else:
        ref_points = context["pareto_front"]
        
    # For each candidate, compute mean squared distance to the nearest front points  
    cand_distances_squared = []
    for i, cand in enumerate(context["pool"]):
        gp = cand["gp_posterior"]
        pred_obj = np.array([gp[name]["mean"] for name in names])
        distances_sq = [np.sum((pred_obj - ref_pt)**2) for ref_pt in ref_points]
        min_dist_squared = min(distances_sq)
        cand_distances_squared.append(min_dist_squared)

    # Normalize inverse front proximity (smaller distance -> higher density -> lower score contribution)
    if len(cand_distances_squared) > 1:
        dists_normed = np.array(cand_distances_squared)
        max_dists, min_dists = np.max(dists_normed), np.min(dists_normed)
        # Avoid division by zero
        front_density_score = (max_dists - dists_normed + 1e-8) / (max_dists - min_dists + 1e-8)
    else:
        front_density_score = [0.5] * len(cand_distances_squared)

    for i, cand in enumerate(context["pool"]):
        gp = cand["gp_posterior"]
        
        # Base acquisition value
        acq_val = cand['acq_value_norm']
        
        # Uncertainty term (normalized by front range)
        sigma_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Combine: high uncertainty, low density -> higher score  
        scores.append(acq_val + 0.5 * sigma_sum * front_density_score[i])

    return scores