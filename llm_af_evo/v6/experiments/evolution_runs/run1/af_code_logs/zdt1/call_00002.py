def modifier(context):
    """Diversity bonus: penalizes candidates similar to high-acquisition neighbors, scaled to [0, 0.3]."""
    pool = context["pool"]
    n_cand = len(pool)
    
    if n_cand <= 1:
        return [0.] * n_cand
    
    # Use x-space for similarity
    X = np.array([cand['x'] for cand in pool])
    
    # Compute pairwise squared Euclidean distances (avoiding self-distances by setting diagonal to inf)
    dists_sq = np.sum((X[:, None] - X[None, :]) ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)

    acq_values = np.array([cand['acq_value_norm'] for cand in pool])
    
    # Identify high-acquisition candidates (top third) to form the reference set
    threshold_acq = np.percentile(acq_values, 70)
    high_idx = np.where(acq_values >= threshold_acq)[0]
    
    values = []
    for i in range(n_cand):
        if len(high_idx) == 1:
            # No other candidates to compare with
            similarity_to_high = 0.0 
        else:  
            dists_to_high = np.min(dists_sq[i, high_idx])
            
            # Kernel function (Gaussian): exp(-d^2 / sigma^2)
            if dists_to_high == np.inf:
                kernel_val = 1e-6
            else:
                # Use a fixed scale for the Gaussian; can tune this value 
                sigma_sq = 0.5  
                similarity_to_high = np.exp(-dists_to_high / sigma_sq) 
            
        values.append(similarity_to_high)
    
    max_value = max(values) if any(v > 1e-6 for v in values) else 1.
        
    # Scale to [0, 0.3]
    scaled_values = [(v/max_value)*0.3 if max_value != 0. else 0. 
                     for v in values]

    return scaled_values