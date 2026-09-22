def modifier(context):
    """Diversity bonus: penalizes candidates similar to high-acquisition neighbors in feature space."""
    if len(context["pool"]) < 2:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    X_features = np.array([cand["x"] for cand in context["pool"]])
    acq_values = np.array([cand["acq_value_norm"] for cand in context["pool"]])

    # Compute pairwise feature distances
    diff_matrix = X_features[:, None, :] - X_features[None, :, :]
    dists = np.sqrt(np.sum(diff_matrix**2, axis=2))

    # Set diagonal to large value so self-similarity is ignored
    np.fill_diagonal(dists, 1e6)

    high_acq_mask = acq_values >= np.percentile(acq_values, max(50 - context["campaign"]["step"] // 3, 10))
    
    values = []
    for i in range(len(context["pool"])):  
        # Get distances to all other candidates
        dists_to_others = dists[i]
        
        # Compute similarity kernel with high-acq neighbors only (excluding self)
        if not np.any(high_acq_mask):
            sim_score = 0.0 
        else:
            similarities = np.exp(-dists_to_others / max(np.std(dists_to_others), 1e-6))
            
            # Only consider similarity to high-acquisition candidates
            similar_scores = similarities * high_acq_mask
            
            if not np.any(similar_scores):
                sim_score = 0.0 
            else:
                total_sim = np.sum(similar_scores)
                
                # Bonus is inversely related to how much this candidate resembles others in the top half  
                norm_factor = max(total_sim, 1e-6) * (np.max(acq_values[high_acq_mask]) - np.min(acq_values))
                sim_score = min(0.3 * total_sim / norm_factor if norm_factor > 0 else 0., 0.3)
        
        values.append(sim_score)

    return values