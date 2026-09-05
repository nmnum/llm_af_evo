def score_pool(context):
    """Score candidates by normalized predicted quality multiplied by inverse diversity; diversity is computed as average similarity to other top-k candidates."""
    names = context["objective_names"]
    
    # Compute raw qualities (sum of means) for each candidate
    qualities = []
    for cand in context["pool"]:
        mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names)
        qualities.append(mu_sum)
        
    min_qual, max_qual = np.min(qualities), np.max(qualities)
    
    # Normalize quality to [0, 1] range
    if max_qual > min_qual:
        norm_quals = (np.array(qualities) - min_qual) / (max_qual - min_qual + 1e-9)
    else:
        norm_quals = np.ones_like(qualities)

    # Compute pairwise similarities in feature space using Gaussian kernel
    X_pool = np.stack([cand["x"] for cand in context["pool"]])
    
    # Pairwise squared Euclidean distances between candidates (excluding self-similarity) 
    dists_sq = ((X_pool[:, None] - X_pool[None, :]) ** 2).sum(axis=2)
  
    # Gaussian kernel similarity
    similarities = np.exp(-dists_sq / (np.var(X_pool, axis=0).mean() + 1e-9))
    
    # Set diagonal to zero so candidates don't penalize themselves 
    np.fill_diagonal(similarities, 0.)
        
    scores = []
    for i in range(len(context["pool"])):
        quality_score = norm_quals[i]
            
        # Compute average similarity with all other points (not including self)
        avg_sim = similarities[i].mean()
      
        diversity_penalty = max(1e-9, 1. - avg_sim) 
        
        final_score = quality_score * diversity_penalty
          
        scores.append(final_score)

    return scores