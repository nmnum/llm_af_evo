def score_pool(context):
    """Blend acquisition value with a coverage-gap score based on distance to nearest Pareto front points."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto_front is too small for k=3 neighbors
    obs = context["Y_obs"] 
    use_pf = len(pf) >= 3
    ref_set = pf if use_pf else obs
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted objective vector (already flipped to maximize)
        pred_obj = np.array([gp[name]["mean"] for name in names])
            
        # Compute mean distance from candidate prediction to k nearest front points
        dists = [np.linalg.norm(pred_obj - ref_point) 
                 for ref_point in ref_set]
        
        if not use_pf:
            # For small fronts, we still want some notion of sparsity,
            # so just sort distances and take the mean over first 3 (or all)
            k = min(3, len(dists))
            sorted_dists = np.sort(np.array(dists))[:k]
        else: 
            # Sort full distance list then get smallest three
            sorted_dists = np.partition(np.array(dists), 2)[:3]

        mean_dist_to_front = np.mean(sorted_dists) if len(sorted_dists) > 0 else float('inf')
        
        # Coverage-gap score is inverse of this: candidates near sparse areas (large distances)
        coverage_gap_score = -mean_dist_to_front
        
        # Blend with acquisition value
        acq_norm = cand["acq_value_norm"]
        blended_score = acq_norm + 1e-3 * coverage_gap_score  
        
        scores.append(blended_score)

    return scores