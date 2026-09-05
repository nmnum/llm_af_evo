def score_pool(context):
    """Blend acquisition value with a coverage-gap term: candidates near sparse front regions get higher scores."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto front is too small for k=3 nearest neighbors
    obs = context["Y_obs"] 
    use_pf = len(pf) >= 3
    points_to_query = pf if use_pf else obs
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Get candidate's predicted objectives (already maximized)
        pred_obj = np.array([gp[name]["mean"] for name in names])
            
        # Compute mean distance to k=3 nearest front points
        if len(points_to_query) == 0:
            coverage_score = 0.0 
        else:  
            distances = [np.linalg.norm(pred_obj - point, ord=2) for point in points_to_query]
            closest_k = np.partition(distances, min(3, len(distances)))[:min(3, len(distances))]
            mean_dist = float(np.mean(closest_k)) if len(closest_k) > 0 else 0.0
            coverage_score = mean_dist
            
        # Blend with acquisition value: higher acq_value + high coverage_gap score 
        base_acq = cand["acq_value_norm"]
        final_score = base_acq + 0.1 * coverage_score  
        
        scores.append(final_score)
    
    return scores