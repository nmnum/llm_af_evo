def score_pool(context):
    """Score candidates by base acquisition value, then greedily reduce scores for proximity to already-picked candidates in feature space."""
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Base scoring using normalized acquisition values (already computed)
    base_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    picked_indices = []
    final_scores = np.zeros(pool_size)

    for step in range(pool_size):
        if len(picked_indices) == 0:
            # Pick the candidate with highest base score
            best_idx = int(np.argmax(base_scores))
        else:
            # Among unpicked candidates, pick one that maximizes (base_score * multiplier)
            max_val = -np.inf
            chosen_idx = None
            
            for i in range(pool_size):
                if i not in picked_indices:
                    base_s = base_scores[i]
                    
                    min_dist_sq = np.inf
                    
                    # Compute squared distance to all already-picked candidates  
                    for pidx in picked_indices:
                        dist_sq = sum((context["pool"][i]["x"][j] - context["pool"][pidx]["x"][j]) ** 2 
                                      for j in range(len(context["pool"][i]["x"])))
                        
                        if dist_sq < min_dist_sq:  
                            min_dist_sq = dist_sq
                    
                    # Multiplier based on inverse exponential of distance
                    multiplier = 1.0 - np.exp(-min_dist_sq)
                    
                    val = base_s * multiplier
                    
                    if val > max_val:
                        max_val = val 
                        chosen_idx = i
            
            best_idx = chosen_idx

        picked_indices.append(best_idx) 
        
        # Assign final score to this candidate
        min_dist_sq = 0.0
        
        for pidx in picked_indices[:-1]:  
            dist_sq = sum((context["pool"][best_idx]["x"][j] - context["pool"][pidx]["x"][j]) ** 2 
                          for j in range(len(context["pool"][best_idx]["x"])))
            
            if dist_sq < min_dist_sq or min_dist_sq == np.inf:
                min_dist_sq = dist_sq
        
        multiplier = 1.0 - np.exp(-min_dist_sq)
        
        final_scores[best_idx] = base_scores[best_idx] * max(multiplier, 0)

    return list(final_scores)