def score_pool(context):
    """Score candidates based on how well they target under-covered regions of the Pareto front using mean nearest-front-distance."""
    names = context["objective_names"]
    
    # Use Y_obs if pareto_front is too small for meaningful density estimation
    pf = context.get("pareto_front", np.array([]))
    if len(pf) < 3:
        reference_points = context["Y_obs"] 
    else:  
        reference_points = pf

    scores = []
    
    # Precompute distances from each candidate to all front points for efficiency
    k_nearest = min(3, len(reference_points)) 
    
    for cand in context["pool"]:
        pred_obj_vector = np.array([cand["gp_posterior"][name]["mean"] 
                                   for name in names])
        
        if len(pf) < 3:
            # Use all observations as reference points
            distances_to_ref = [np.linalg.norm(pred_obj_vector - obs_vec)
                                for obs_vec in context['Y_obs']]
            
        else:  
             # Compute Euclidean distance to each point on the front 
            distances_to_ref = []
            for ref_point in pf:
                dist = np.sqrt(sum((pred_obj_vector[i] - ref_point[i])**2
                                   for i in range(len(names))))
                distances_to_ref.append(dist)
                
        nearest_distances = sorted(distances_to_ref)[:k_nearest]
        
        # Mean of k-nearest distances to front points  
        coverage_gap_score = sum(nearest_distances) / len(nearest_distances)

        acq_value_norm = cand["acq_value_norm"]
 
        blended_score = 0.95 * acq_value_norm + (1 - 0.95) * coverage_gap_score
        
        scores.append(blended_score)
        
    return scores