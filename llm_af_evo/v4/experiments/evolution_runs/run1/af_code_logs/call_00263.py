def score_pool(context):
    """Rank candidates by a combination of normalized quality (predicted objective sum) and joint diversity; higher is better."""
    names = context["objective_names"]
    
    # Compute raw qualities as summed means
    qualities = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        mu_sum = sum(gp[name]["mean"] for name in names)
        qualities.append(mu_sum)

    min_qual, max_qual = np.min(qualities), np.max(qualities)  
    if abs(max_qual - min_qual) < 1e-9:
        normed_quals = [0.0] * len(qualities)
    else:
        normed_quals = [(q - min_qual) / (max_qual - min_qual + 1e-9) for q in qualities]

    # Build similarity kernel based on feature space distances
    X_pool = np.array([cand["x"] for cand in context["pool"]])
    
    # Compute pairwise squared Euclidean distances between candidates  
    dists_sq = np.sum((X_pool[:, None] - X_pool[None, :]) ** 2, axis=-1)
   
    # Convert to similarity using Gaussian kernel
    sigma = np.median(dists_sq[dists_sq > 0])
    if sigma == 0:
        sim_matrix = np.eye(len(context["pool"]))
    else: 
        K = np.exp(-dists_sq / (2 * sigma ** 2))
        
        # Remove self-similarity from averaging
        for i in range(K.shape[0]):
            K[i, i] = 1e-9
            
        sim_matrix = K
    
    scores = []
    
    for idx in range(len(context["pool"])):
        qual_score = normed_quals[idx]
        
        # Compute diversity as average similarity to all other candidates
        avg_sim_to_others = np.mean(sim_matrix[idx, :])
 
        div_score = 1.0 - avg_sim_to_others
        
        combined_score = qual_score * div_score

        scores.append(combined_score)
    
    return scores