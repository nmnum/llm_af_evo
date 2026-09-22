def score_pool(context):
    """Estimate each candidate’s potential to expand hypervolume while penalizing similarity to top-ranked picks."""
    names = context["objective_names"]
    
    # Compute acquisition scores and track top candidates by acq_value_norm
    pool_with_scores = []
    for cand in context["pool"]:
        gp_mean = [cand["gp_posterior"][name]["mean"] for name in names]
        acq_score = cand["acq_value_norm"]
        pool_with_scores.append((cand, gp_mean, acq_score))

    # Sort by acquisition value (descending)
    sorted_candidates = sorted(pool_with_scores, key=lambda x: x[2], reverse=True)

    scores = []
    
    for i, (cand, mean_vals, _) in enumerate(sorted_candidates):
        base_acq_value_norm = cand["acq_value_norm"]
        
        # Compute distance to top 3 candidates by acquisition value
        repulsion_penalty = 0.0
        
        if len(sorted_candidates) > 1:
            for j in range(min(3, i)):
                other_mean_vals = sorted_candidates[j][1]
                dist_to_top = np.linalg.norm(np.array(mean_vals) - np.array(other_mean_vals))
                
                # Normalize by objective ranges to avoid bias from scale
                norm_factor = sum(context["pareto_front_range"][name] for name in names)
                if not (norm_factor == 0):
                    repulsion_penalty += dist_to_top / norm_factor
                    
        final_score = base_acq_value_norm - 0.1 * repulsion_penalty  
        
        scores.append(final_score)

    # Return original order
    ordered_scores = [None] * len(context["pool"])
    
    for i, (cand_orig_idx, _, _) in enumerate([(j, sorted_candidates[j][2]) 
                                                for j in range(len(sorted_candidates))]):
        ordered_scores[cand_orig_idx] = scores[i]
        
    return ordered_scores