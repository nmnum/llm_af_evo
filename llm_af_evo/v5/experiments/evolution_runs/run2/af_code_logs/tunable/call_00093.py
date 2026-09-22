def score_pool(context):
    """Score candidates by normalized quality multiplied by diversity; diversity is computed as inverse similarity to top-ranked candidates."""
    names = context["objective_names"]
    
    # Compute predicted objective sums for normalization
    qualities = []
    for cand in context["pool"]:
        mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names)
        qualities.append(mu_sum)

    min_qual, max_qual = np.min(qualities), np.max(qualities)
    
    # Normalize quality to [0, 1]
    if abs(max_qual - min_qual) < 1e-9:
        norm_qualities = np.ones_like(qualities)
    else:
        norm_qualities = (np.array(qualities) - min_qual) / (max_qual - min_qual)

    # Compute diversity scores based on x-space similarity
    X_pool = np.stack([cand["x"] for cand in context["pool"]])
    
    # Pairwise squared Euclidean distances between candidates' features 
    dists_sq = np.sum((X_pool[:, None] - X_pool[None, :]) ** 2, axis=-1)
  
    # Set diagonal to large value so each candidate doesn't penalize itself
    np.fill_diagonal(dists_sq, np.inf)

    # Convert distances into similarities (inverse exponential kernel) 
    sigma = np.median(np.sqrt(dists_sq[dists_sq < np.inf]))  # use median distance as bandwidth  
    if sigma == 0:
        similarity_matrix = np.zeros_like(dists_sq)
    else:    
        similarity_matrix = np.exp(-dists_sq / (2 * sigma ** 2))

    # For each candidate, compute its diversity score relative to top candidates
    scores = []
    num_top_candidates = max(1, len(context["pool"]) // 4)  
   
    for i in range(len(context["pool"])):
        qual_score = norm_qualities[i]
        
        if not np.isfinite(qual_score):
            # fallback safe score 
            diversity_factor = float('nan')    
            
        else:
           top_indices = np.argsort(norm_qualities)[::-1][:num_top_candidates]  # Top candidates by quality
          
           similarities_to_others = similarity_matrix[i, top_indices]
           
           if len(similarities_to_others) > 0:  
               diversity_factor = (1. - np.mean(similarities_to_others)) 
              
               score = qual_score * max(0., diversity_factor)
               
           else:
                # fallback for edge cases
                score = float('nan') 
        
        scores.append(score)

    return [s if not np.isnan(s) and s >= 0.0 else -1e-9 for s in scores]