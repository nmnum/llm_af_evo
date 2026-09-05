def score_pool(context):
    """Integrate acquisition value with inverse uncertainty scaled by front density to bias selection toward underexplored yet promising regions."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto_front is too small for k=3 nearest neighbors
    use_y_obs = len(pf) < 3
    
    scores = []
    for cand in context["pool"]:
        gp_mean = [cand["gp_posterior"][name]["mean"] for name in names]
        
        # Compute mean distance to k-nearest front points (or Y_obs if too few)
        distances = []
        reference_points = pf if not use_y_obs else context["Y_obs"]
        cand_point = np.array(gp_mean)

        for ref_point in reference_points:
            dist = np.linalg.norm(cand_point - ref_point, ord=2)
            distances.append(dist)
        
        # Get k nearest (k=3) 
        sorted_distances = sorted(distances)[:min(3, len(distances))]
        mean_dist_to_front = sum(sorted_distances)/len(sorted_distances)

        acq_value_norm = cand["acq_value_norm"]
        sigma_sum = sum(cand["gp_posterior"][name]["std"] for name in names)
        
        # Scale uncertainty by inverse of front density (closer to front is less dense, more uncertain regions are harder explored) 
        if use_y_obs:
            inv_density_factor = 1.0 / mean_dist_to_front
        else:  
            range_0 = context["pareto_front_range"][names[0]]
            # Normalize distance and invert for density (smaller dist -> higher density)
            normalized_distance = min(1., max(0., mean_dist_to_front / range_0))
            inv_density_factor = 1. - normalized_distance
            
        final_score = acq_value_norm + sigma_sum * inv_density_factor
        scores.append(final_score)

    return scores