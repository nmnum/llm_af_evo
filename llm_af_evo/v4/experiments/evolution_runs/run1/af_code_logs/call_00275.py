def score_pool(context):
    """Rank candidates greedily by base acquisition value, then suppress scores of similar remaining candidates based on Euclidean distance in feature space."""
    if not context["pool"]:
        return []
    
    names = context["objective_names"]
    pool_size = len(context["pool"])
    selected_indices = set()
    final_scores = [0.0] * pool_size
    base_scores = [cand['acq_value_norm'] for cand in context["pool"]]
    
    # Greedily select candidates and suppress similar ones
    while len(selected_indices) < pool_size:
        best_idx = -1
        best_score = float('-inf')
        
        # Find the highest-scoring candidate not yet selected
        for i, score in enumerate(base_scores):
            if i not in selected_indices and score > best_score:
                best_score = score
                best_idx = i
                
        assert best_idx != -1
        
        final_scores[best_idx] = base_scores[best_idx]
        selected_indices.add(best_idx)
        
        # Suppress remaining candidates based on feature space distance to the picked one
        if len(selected_indices) < pool_size:
            x_picked = context["pool"][best_idx]["x"]
            
            for i in range(pool_size):
                if i not in selected_indices and i != best_idx:
                    dist_squared = np.sum((context["pool"][i]["x"] - x_picked)**2)
                    
                    # Compute multiplier as 1.0 - exp(-dist), where distance is sqrt of squared
                    multipler = 1.0 - np.exp(-np.sqrt(dist_squared))
                    assert multipler >= 0
                    
                    base_scores[i] *= multipler
    
    return final_scores