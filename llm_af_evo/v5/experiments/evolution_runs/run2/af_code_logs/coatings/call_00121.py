def score_pool(context):
    """Blend hypervolume acquisition with a coverage-gap term targeting sparse Pareto front regions."""
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

        # Blend with acquisition value
        acq_value_norm = cand["acq_value_norm"]
        coverage_gap_score = 1.0 - (mean_dist_to_front / context["pareto_front_range"][names[0]]) if not use_y_obs else mean_dist_to_front
        
        final_score = acq_value_norm + 0.2126 * coverage_gap_score
        scores.append(final_score)
        
    return scores