def score_pool(context):
    """Score candidates by expected hypervolume contribution adjusted for proximity-based redundancy suppression."""
    names = context["objective_names"]
    
    # Use acquisition value as base quality metric
    acq_values = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    
    # Compute feature distances between all candidates  
    X = np.stack([cand["x"] for cand in context["pool"]]) 
    dists_sq = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)
    
    # Identify near-duplicates (within 1% feature distance) and suppress them
    epsilon = 0.01  
    dup_mask = dists_sq < (epsilon**2)
    np.fill_diagonal(dup_mask, False)

    scores = []
    for i in range(len(context["pool"])):
        # Start with acquisition value 
        score = acq_values[i]
        
        # Reduce score if this candidate is too similar to a higher-ranked one
        neighbors = dup_mask[i]  
        if np.any(neighbors):
            # Find the highest-scoring neighbor (i.e., already selected)
            neighbor_scores = acq_values[neighbors]
            best_neighbor_score = np.max(neighbor_scores) 
            
            # Suppress this candidate's score based on how much better its duplicate is
            suppression_factor = 0.5 * max(0, best_neighbor_score - score)  
            score -= suppression_factor
            
        scores.append(score)
    
    return scores