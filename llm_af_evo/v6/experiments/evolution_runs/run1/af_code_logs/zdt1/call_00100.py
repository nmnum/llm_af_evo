def modifier(context):
    """Diversity bonus: penalizes candidates similar to high-acquisition-score peers in feature space."""
    if len(context["pool"]) < 2:
        return [0.0] * len(context["pool"])
    
    # Use normalized x features for similarity calculation
    X = np.array([cand['x'] for cand in context["pool"]])
    
    # Compute pairwise Euclidean distances between all candidates (excluding self)
    diff_matrix = X[:, None, :] - X[None, :, :]
    dists = np.sqrt(np.sum(diff_matrix**2, axis=2))
    
    # Set diagonal to large value so similarity is 0 for each candidate vs itself
    np.fill_diagonal(dists, np.inf)

    acq_values = [cand['acq_value_norm'] for cand in context["pool"]]
    high_acq_mask = np.array(acq_values) > np.percentile(acq_values, 75)
    
    bonus_factors = []
    max_bonus = 0.3
    
    # For each candidate
    for i, (dist_row, is_high_acq) in enumerate(zip(dists[i], high_acq_mask)):
        if not is_high_acq:
            bonus_factors.append(0.)
            continue
            
        # Find indices of candidates with both high acquisition value and low distance to this one  
        close_high = np.where((dists[i] < 1e-5) & (high_acq_mask))[0]
        
        # If no similar peers, full reward; if many similar ones in top quartile,
        # reduce bonus
        n_similar_peers_in_top_quartile = len(close_high)
        similarity_factor = np.exp(-n_similar_peers_in_top_quartile * 2.5) 
                
        bonus_factors.append(similarity_factor * max_bonus)

    return bonus_factors