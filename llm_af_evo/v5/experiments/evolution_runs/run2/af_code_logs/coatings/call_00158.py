def score_pool(context):
    """Integrate hypervolume acquisition with an uncertainty-weighted frontier coverage signal that adapts its focus based on progress and stagnation."""
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
        uncertainty = np.mean([cand["gp_posterior"][name]["std"] for name in names])
        
        # Dynamic weight based on campaign progress and stagnation
        p = context["campaign"]["progress"]
        s = context["campaign"]["stagnant_batches"]
        w_exploit = 1.0 - min(p, 1.0) * (1 + 0.5*s)
        w_uncertain = max(0., 2.*p*(s+3.) / (4.+s))
        
        # Blend acquisition with a coverage term weighted by uncertainty
        if not use_y_obs:
            coverage_gap_score = 1.0 - mean_dist_to_front / context["pareto_front_range"][names[0]]
        else:
            coverage_gap_score = -(mean_dist_to_front) 

        final_score = w_exploit * acq_value_norm + (w_uncertain * uncertainty + (1-w_uncertain)*coverage_gap_score)
        
        scores.append(final_score)

    return scores