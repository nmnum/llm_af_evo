def score_pool(context):
    """Suppress scores of candidates that are too similar to top-ranked ones, encouraging diversity in selected batch."""
    names = context["objective_names"]
    
    # Start with acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Rank candidates by their acquisition score
    ranked_indices = np.argsort(acq_scores)[::-1]
    
    # Initialize suppression mask (True means suppress this candidate)
    to_suppress = [False] * len(context["pool"])
    
    # Define how many top candidates we consider for diversity
    num_top_candidates = min(5, int(len(context["pool"]) / 2))
        
    if num_top_candidates > 0:
        front_range = context["pareto_front_range"]
            
        # For each candidate in the pool (in ranked order)
        for i_idx in range(min(num_top_candidates, len(ranked_indices))):
            idx1 = ranked_indices[i_idx]
                
            cand_x_1 = context['pool'][idx1]["x"] 
                    
            if to_suppress[idx1]:
                continue
                    
            # Compare against other candidates that are not yet suppressed
            for j in range(i_idx + 1, len(ranked_indices)):
                idx2 = ranked_indices[j]
                        
                if to_suppress[idx2]:  
                    continue
                        
                cand_x_2 = context['pool'][idx2]["x"]
                
                # Compute normalized Euclidean distance between candidates 
                dist_squared = np.sum((cand_x_1 - cand_x_2)**2)
                    
                # Normalize by feature dimensionality
                norm_dist_sq = dist_squared / len(cand_x_1)  
                        
                if norm_dist_sq < 0.05:   # If very close in input space, suppress the lower acquisition one 
                    to_suppress[idx2] = True
                    
    final_scores = acq_scores.copy()
    
    for i, shouldSuppress in enumerate(to_suppress):
        if shouldSuppress:
            final_scores[i] -= (final_scores[i]/10.)  # Reduce score by small fraction
            
    return list(final_scores)