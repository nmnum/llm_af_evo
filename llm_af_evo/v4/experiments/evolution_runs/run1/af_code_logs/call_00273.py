def score_pool(context):
    """Use acquisition value as primary signal and augment with uncertainty-normalized distance to Pareto front for diversity."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute distances from each candidate's predicted means to the reference point
    cand_means = []
    for cand in context["pool"]:
        mean_vals = [cand["gp_posterior"][name]["mean"] for name in names]
        cand_means.append(mean_vals)
        
    dists_to_ref = np.linalg.norm(np.array(cand_means) - ref_point, axis=1)

    # Normalize distances to 0-1 scale
    max_dist = np.max(dists_to_ref)
    
    scores = []
    for i, (cand, dist) in enumerate(zip(context["pool"], dists_to_ref)):
        acq_val_norm = cand["acq_value_norm"]
        
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / context["pareto_front_range"][name] 
                        for name in names)
        
        # Use uncertainty to adjust the distance-based novelty
        normalized_dist = dist / max_dist if max_dist > 0 else 0.0
        
        score = acq_val_norm + 0.1 * sigma_sum - 0.2 * (normalized_dist ** 2)

        scores.append(score)
    
    return scores