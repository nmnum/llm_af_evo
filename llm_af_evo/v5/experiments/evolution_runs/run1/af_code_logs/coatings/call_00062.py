def score_pool(context):
    """Blend hypervolume improvement with a coverage-gap term targeting sparse front regions."""
    import numpy as np
    
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto front is too small for k=3 neighbors
    obs_for_dist = pf if len(pf) >= 3 else context["Y_obs"] 
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        pred_obj_vec = np.array([gp[name]["mean"] for name in names])
        
        # Compute distances to k=3 nearest front points
        dists_to_front = [np.linalg.norm(pred_obj_vec - point) for point in obs_for_dist]
        dists_sorted = sorted(dists_to_front)
        mean_nearest_dist = np.mean(dists_sorted[:min(3, len(obs_for_dist))])
        
        # Blend with acquisition value (smaller weight as secondary term)
        acq_val_norm = cand["acq_value_norm"]
        coverage_gap_score = 1.0 - mean_nearest_dist
        final_score = acq_val_norm + 0.2 * coverage_gap_score
        
        scores.append(final_score)

    return scores