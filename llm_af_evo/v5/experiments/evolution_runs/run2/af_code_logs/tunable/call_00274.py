def score_pool(context):
    """Integrate acquisition value with dynamic uncertainty-weighted regret from Pareto front boundaries."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto_front is too small for k=3 nearest neighbors  
    use_y_obs = len(pf) < 3
    reference_points = pf if not use_y_obs else context["Y_obs"] 
    
    scores = []
    for cand in context["pool"]:
        gp_mean = [cand["gp_posterior"][name]["mean"] for name in names]
        
        # Compute mean distance to k-nearest front points (or Y_obs)
        distances = []
        cand_point = np.array(gp_mean)

        for ref_point in reference_points:
            dist = np.linalg.norm(cand_point - ref_point, ord=2)  
            distances.append(dist)
            
        sorted_distances = sorted(distances)[:min(3, len(distances))]
        mean_dist_to_front = sum(sorted_distances)/len(sorted_distances) if sorted_distances else 0.0

        # Normalize distance by front range for score
        norm_distance = (mean_dist_to_front / context["pareto_front_range"][names[0]]) if not use_y_obs and context["pareto_front_range"][names[0]] > 0 else mean_dist_to_front
        
        acq_value_norm = cand["acq_value_norm"]
        
        # Compute uncertainty-weighted regret score
        sigma_sum = sum(cand["gp_posterior"][name]["std"] for name in names)
        if context['campaign']['progress'] < 0.3:
            # Early exploration: favor higher acquisition + lower distance to front (more unexplored regions) 
            final_score = acq_value_norm * (1 - norm_distance / 2.)
        else:
            # Later exploitation: blend with uncertainty, but heavily prioritize high hypervolume gain
            regretness = sigma_sum * max(0., 1.5 - context['campaign']['progress'])  
            final_score = acq_value_norm + 0.3*(regretness) 
        scores.append(final_score)
        
    return scores