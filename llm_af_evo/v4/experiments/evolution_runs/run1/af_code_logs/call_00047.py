def score_pool(context):
    """Rank candidates by joint diversity-aware quality: normalize predicted objective sum to non-negative values then multiply by inverse similarity to top-ranked candidates."""
    names = context["objective_names"]
    
    # Compute raw qualities (sum of means) for each candidate
    qualities = []
    for cand in context["pool"]:
        mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names)
        qualities.append(mu_sum)
        
   qualities = np.array(qualities)
    
    # Normalize quality to [0, 1] range (non-negative)
    q_min, q_max = qualities.min(), qualities.max()
    if q_max > q_min:
        norm_qualities = (qualities - q_min) / (q_max - q_min + 1e-9)
    else:
        norm_qualities = np.zeros_like(qualities)

    # Compute similarity matrix based on feature space distances
    X_pool = np.array([cand["x"] for cand in context["pool"]])
    
    # Pairwise squared Euclidean distances (excluding self-similarity by setting diagonal to inf)
    dists_sq = ((X_pool[:, None] - X_pool[None, :]) ** 2).sum(axis=2) 
    np.fill_diagonal(dists_sq, np.inf)

    # Compute similarity matrix using Gaussian kernel
    sigma = np.median(np.sqrt(dists_sq[np.isfinite(dists_sq)]))
    if sigma == 0:
        similarities = np.ones_like(dists_sq)
    else:  
        similarities = np.exp(-dists_sq / (2 * sigma ** 2))

    # Set diagonal to zero for self-similarity
    np.fill_diagonal(similarities, 0)

    scores = []
    
    # For each candidate compute its diversity score relative to already selected top candidates 
    sorted_indices = np.argsort(norm_qualities)[::-1]
    
    selected = set()
    for i in range(len(context["pool"])):
        cand_idx = sorted_indices[i] 
        
        if len(selected) == 0:
            div_score = 1.0
        else:  
            # Compute average similarity to already-selected candidates 
            sim_to_selected = np.mean(similarities[cand_idx][list(selected)])
            div_score = max(0, 1 - sim_to_selected)
            
        scores.append(norm_qualities[i] * div_score)    
        
        selected.add(cand_idx)

    return list(scores[::-1]) # Return in original order