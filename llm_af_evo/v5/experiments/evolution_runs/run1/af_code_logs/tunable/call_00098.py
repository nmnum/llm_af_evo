def score_pool(context):
    """Score by acquisition value adjusted for diversity: penalize candidates near top-ranked ones."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute scores using only acq_value_norm as base
    raw_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Identify the top 30% of candidates by acquisition value (or at least 1)
    n_top = max(1, len(context["pool"]) // 3)
    sorted_indices = np.argsort(raw_scores)[::-1][:n_top]
    top_candidates = [context["pool"][i] for i in sorted_indices]

    # For each candidate, compute average inverse distance to the top candidates
    scores = []
    for cand in context["pool"]:
        x_cand = cand["x"]
        
        distances_to_top = [
            np.linalg.norm(x_cand - tc["x"]) / max(1e-8, front_range[name])
            for name in names 
            for tc in top_candidates
        ]
            
        # Inverse of mean distance (higher means farther from the crowd)
        if len(distances_to_top) == 0:
            inv_mean_dist = 0.0  
        else:    
            mean_distance = np.mean(distances_to_top)
            inv_mean_dist = 1. / max(1e-8, mean_distance)

        # Final score is acquisition value multiplied by inverse distance bonus
        scores.append(raw_scores[context["pool"].index(cand)] * (0.5 + 2.0 * inv_mean_dist))

    return scores