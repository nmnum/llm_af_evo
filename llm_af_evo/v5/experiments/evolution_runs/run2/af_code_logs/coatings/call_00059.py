def score_pool(context):
    """Blend acquisition value with a coverage-gap score based on distance to nearest Pareto front points."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    y_obs = context["Y_obs"] 
    k = 3
    
    # Use Y_obs if pareto_front is too small
    use_yobs = len(pf) < k
    ref_set = y_obs if use_yobs else pf

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        pred_obj = np.array([gp[name]["mean"] for name in names])
    
        # Compute distances to reference set (front or obs)
        dists = [np.linalg.norm(pred_obj - ref_pt) for ref_pt in ref_set]
        dists.sort()
            
        mean_dist_to_front = sum(dists[:k]) / k
        
        blended_score = cand["acq_value_norm"] + 0.1 * mean_dist_to_front
        scores.append(blended_score)
    
    return scores