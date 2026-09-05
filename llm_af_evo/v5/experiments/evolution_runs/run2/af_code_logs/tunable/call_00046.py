def score_pool(context):
    """Suppress scores of nearby candidates after greedily selecting the highest-acquisition one, promoting diversity."""
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Base acquisition values (already hypervolume improvement estimates)
    base_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    picked_indices = []
    final_scores = np.zeros(pool_size)

    while len(picked_indices) < pool_size:
        if notpicked := [i for i in range(pool_size) if i not in picked_indices]:
            # Pick the best unpicked candidate
            idx = notpicked[np.argmax(base_scores[notpicked])]
            
            # Assign its final score (base_score * multiplier)
            base_val = base_scores[idx]
            final_scores[idx] = base_val
            
            # Mark as picked for next iteration's distance calculations  
            picked_indices.append(idx)

    return list(final_scores)