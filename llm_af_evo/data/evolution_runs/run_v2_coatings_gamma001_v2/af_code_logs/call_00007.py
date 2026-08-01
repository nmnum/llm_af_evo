def score_pool(context):
    """Rank candidates greedily by base score, then suppress remaining candidates based on proximity to already-picked ones."""
    names = context["objective_names"]
    pool = context["pool"]
    scores = []
    picked = []
    
    # Compute base scores for all candidates
    base_scores = []
    for cand in pool:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        base_scores.append(mu_sum)
    
    # Greedily pick candidates
    remaining = list(range(len(pool)))
    while remaining:
        # Pick the best among remaining
        best_idx = remaining[base_scores[remaining[0]] if len(remaining) == 1 else max(remaining, key=lambda i: base_scores[i])]
        picked.append(best_idx)
        remaining.remove(best_idx)
        
        # Suppress scores of remaining candidates based on proximity to picked ones
        for idx in remaining:
            cand = pool[idx]
            min_dist_x = float('inf')
            min_dist_obj = float('inf')
            
            for picked_idx in picked:
                picked_cand = pool[picked_idx]
                
                # Distance in feature space
                dist_x = np.linalg.norm(cand["x"] - picked_cand["x"])
                min_dist_x = min(min_dist_x, dist_x)
                
                # Distance in objective space
                dist_obj = sum(abs(cand["gp_posterior"][name]["mean"] - picked_cand["gp_posterior"][name]["mean"]) for name in names)
                min_dist_obj = min(min_dist_obj, dist_obj)
            
            # Use the minimum of both distances as a proxy for closeness
            radius = min_dist_x + min_dist_obj  # Simple heuristic to combine x and objective space distances
            
            # Apply suppression factor: 1.0 - exp(-radius) or similar
            penalty_factor = 1.0 - np.exp(-radius)
            
            # Reduce the base score by this factor
            scores[idx] = base_scores[idx] * (1.0 - penalty_factor)
    
    # Fill in scores for picked candidates with their base scores
    final_scores = [0.0] * len(pool)
    for i, idx in enumerate(picked):
        final_scores[idx] = base_scores[idx]
    
    return final_scores