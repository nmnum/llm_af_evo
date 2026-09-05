def score_pool(context):
    """Score candidates based on acquisition value blended with a coverage-gap term targeting sparse Pareto front regions."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Use Y_obs if pareto_front is too small for k=3 nearest neighbors
    pf = context.get("pareto_front", None)
    y_obs = context.get("Y_obs")
    if pf is not None and len(pf) >= 3:
        front_points = pf.copy()
    else:
        # Fall back to observed points when the Pareto front is too small
        front_points = np.array(y_obs).copy() 
    
    scores = []
    
    for cand in context["pool"]:
        
        gp_posterior = cand['gp_posterior']
        predicted_objectives = [gp_posterior[name]["mean"] for name in names]
                
        # Compute distances from candidate to all points on the front
        dists_to_front = np.linalg.norm(front_points - predicted_objectives, axis=1)
            
        sorted_dists = np.sort(dists_to_front)        
        k_nearest_distances = sorted_dists[:3] if len(sorted_dists) >= 3 else sorted_dists
        
        # Mean of the nearest distances to front points
        coverage_gap_score = float(np.mean(k_nearest_distances))
                
        acquisition_value_norm = cand["acq_value_norm"]
        
        blended_score = acquisition_value_norm + (0.1 * coverage_gap_score)
            
        scores.append(blended_score)

    return scores