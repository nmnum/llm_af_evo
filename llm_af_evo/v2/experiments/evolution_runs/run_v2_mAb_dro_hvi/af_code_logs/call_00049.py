def score_pool(context):
    """
    Exploitation with uncertainty-aware diversity: rank by predicted mean sum,
    but reduce scores for candidates that are close (in feature space) to 
    already-selected high-scoring ones, encouraging exploration of diverse regions.
    This avoids the pure exploitation overfitting while maintaining strong
    performance through a mechanism inspired by greedy batch selection strategies.  
    """
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Get scores for all candidates based on mean prediction (exploitation)
    raw_scores = [sum(cand["gp_posterior"][name]["mean"] for name in names) 
                  for cand in context["pool"]]
    
    if not raw_scores:
        return []
        
    top_k_indices = np.argsort(raw_scores)[-min(5, pool_size):][::-1]  # Top-5 indices
    
    scores = [0.0] * len(context["pool"])
    
    selected_features = []  
    for i in top_k_indices: 
        cand_x = context["pool"][i]["x"]
        
        if not selected_features:
            # First candidate gets full score
            scores[i] = raw_scores[i]
            selected_features.append(cand_x)
        else:
            min_dist_to_selected = np.inf
            
            for sel_feat in selected_features: 
                dist = np.linalg.norm(sel_feat - cand_x)  
                
                if dist < min_dist_to_selected:
                    min_dist_to_selected = dist
                    
            # Reduce score based on inverse distance to nearest already-selected candidate
            diversity_penalty_factor = 1.0 / (min_dist_to_selected + 1e-8)
            
            scores[i] = raw_scores[i] * np.exp(-diversity_penalty_factor) 
                
            selected_features.append(cand_x)

    return scores