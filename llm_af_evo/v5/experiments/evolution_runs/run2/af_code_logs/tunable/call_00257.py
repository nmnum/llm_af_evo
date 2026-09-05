def score_pool(context):
    """Blend acquisition value with uncertainty-aware progress tracking to favor candidates near underexplored front regions."""
    names = context["objective_names"]
    
    scores = []
    for cand in context["pool"]:
        gp_mean = [cand["gp_posterior"][name]["mean"] for name in names]
        
        # Compute distance from candidate mean to nearest Pareto point
        pf_distances = [
            np.linalg.norm(np.array(gp_mean) - ref_point)
            for ref_point in context["pareto_front"]
        ]
        min_pf_dist = min(pf_distances) if pf_distances else float('inf')
        
        # Normalize the minimum front distance by range of first objective 
        norm_min_dist = min_pf_dist / context["pareto_front_range"][names[0]]
        
        acq_value_norm = cand["acq_value_norm"]
        
        # Use inverse of normalized distance as a progress signal: candidates near sparse regions get higher score
        if norm_min_dist == 0:
            progress_signal = float('inf')   # This candidate is on the front, so it's maximally progressive 
        else:
            progress_signal = 1. / (norm_min_dist + 1e-8)    # Avoid division by zero
            
        # Scale uncertainty based on how close we are to current pareto frontier
        gp_std = [cand["gp_posterior"][name]["std"] for name in names]
        
        std_sum_normed = sum(gp_std)/len(names)
        if not context["pareto_front"].size:
            normalized_uncertainty = 1.0   # No front yet, so be exploratory
        else:  
            range_scale_factor = min(context["pareto_front_range"][name] for name in names) 
            normalized_uncertainty = std_sum_normed / (range_scale_factor + 1e-8)
            
        
        final_score = acq_value_norm * progress_signal - 0.5 * normalized_uncertainty
        scores.append(final_score)

    return scores