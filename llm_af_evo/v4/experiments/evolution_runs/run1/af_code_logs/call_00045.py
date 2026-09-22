def score_pool(context):
    """Rank candidates by joint diversity-aware quality: normalize predicted objective sum to non-negative values then multiply by inverse similarity to high-scoring candidates."""
    names = context["objective_names"]
    
    # Extract predictions and compute raw qualities (sum of means)
    qualities = []
    x_features = []
    obj_predictions = []

    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        qualities.append(mu_sum)
        x_features.append(cand["x"])
        
        # Build objective prediction vector
        pred_obj_vec = np.array([gp[name]["mean"] for name in names])
        obj_predictions.append(pred_obj_vec)

    # Normalize quality to [0, 1] range (non-negative)
    min_qual = float(np.min(qualities))
    max_qual = float(np.max(qualities))
    
    if abs(max_qual - min_qual) < 1e-9:
        normed_quals = np.ones(len(qualities)) 
    else:  
        normed_quals = (np.array(qualities) - min_qual) / (max_qual - min_qual)
        
    # Compute pairwise similarities in x-space using Gaussian kernel
    X = np.stack(x_features, axis=0)  # shape [n_pool, 6]
    
    # Pairwise squared Euclidean distances: D[i,j] = ||x_i - x_j||^2  
    diff_sq = (X[:, None, :] - X[None, :, :]) ** 2
    dists_sq = np.sum(diff_sq, axis=2) 
    
    # Gaussian kernel similarity matrix S[i,j]
    sigma_x = np.median(dists_sq.flatten()) 
    if abs(sigma_x) < 1e-9:
        sim_matrix = np.eye(len(qualities))  
    else:    
        K_sim = np.exp(-dists_sq / (2 * sigma_x))
        
        # Set diagonal to zero since we don't want self-similarity
        np.fill_diagonal(K_sim, 0)
            
        # Normalise rows so they sum up to one for averaging later 
        row_sums = K_sim.sum(axis=1) + 1e-9  
        sim_matrix = (K_sim.T / row_sums).T

    scores = []
    
    for i in range(len(context["pool"])):
        
        # Get normalized quality score
        qual_score = normed_quals[i]
                
        if len(scores) == 0:
            diversity_penalty = 1.0  
        else: 
            prev_selected_idxes = np.argsort(-np.array(scores))[:min(5, len(context["pool"]))] 
            
            # Compute average similarity to previously selected candidates
            avg_sim_to_prev = sum(sim_matrix[i][j]*normed_quals[j]/row_sums[j]
                                  for j in prev_selected_idxes 
                                  if normed_quals[j]>0) / (len(prev_selected_idxes)+1e-9)
            
            # Diversity is inverse of average similarity  
            diversity_score = 1. - avg_sim_to_prev
            
        final_score = qual_score * max(0., diversity_score) 
        
        scores.append(final_score)

    return [float(s) for s in scores]