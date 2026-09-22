def score_pool(context):
    """Estimate candidate dominance potential via noisy GP sampling and blend with acquisition value."""
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
        
        # Estimate dominance potential via noisy sampling
        n_samples = 10
        dominated_count = 0
        
        for _ in range(n_samples):
            sampled_means = []
            gp_posterior = cand["gp_posterior"] 
            for name in names:
                mean_val, std_val = gp_posterior[name]["mean"], gp_posterior[name]["std"]
                sample = np.random.normal(mean_val, std_val)
                sampled_means.append(sample) 
            
            # Check if this noisy sample is dominated by current front
            cand_sampled = np.array(sampled_means)
            
            dominates_any = False 
            for ref_point in reference_points:
                # Candidate dominates point if all objectives are >= and at least one strict >
                if all(cand_sampled[i] >= ref_point[i] for i in range(len(names))) and \
                   any(cand_sampled[i] > ref_point[i] for i in range(len(names))):
                    dominates_any = True
                    break
            
            # If sampled point is dominated by current front, count it  
            if not dominates_any:
                dominated_count += 1
                
        dominance_ratio = dominated_count / n_samples
        
        final_score = acq_value_norm + (0.2 * (1 - dominance_ratio)) 
        scores.append(final_score)
        
    return scores