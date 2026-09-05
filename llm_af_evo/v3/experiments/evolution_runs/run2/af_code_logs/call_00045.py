def score_pool(context):
    """Scores candidates based on hypervolume potential adjusted by repulsive forces from already selected high-scoring points."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    
    # Compute base scores using normalized mean for exploitation and uncertainty
    base_scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]        
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names) 
        score = 0.5 * mu_sum_norm + 0.5 * sigma_norm
        base_scores.append(score)

    # Apply repulsion from top candidates (those already selected or highly ranked previously)
    sorted_indices = np.argsort(base_scores)[::-1]
    
    # Only consider the highest scoring half for suppression effects 
    num_suppressors = max(2, len(context["pool"]) // 4)  
    suppressor_ids = set(sorted_indices[:num_suppressors])
        
    final_scores = []
    for i, cand in enumerate(context["pool"]):
        score = base_scores[i]
                
        # Apply repulsion if candidate is near any top-ranked one
        x_cand = cand["x"]
        penalize = False
        
        for j in suppressor_ids:
            if i == j: continue  # Skip self
            
            x_suppressor = context["pool"][j]["x"] 
            dist = np.linalg.norm(x_cand - x_suppressor)
            
            # Suppress candidates that are too close to high-scoring ones
            if dist < (0.1 * max(front_range.values())) and score > 0:
                penalize = True  
                
        if penalize:   
            score *= 0.8   # Reduce the score by a fixed factor
            
        final_scores.append(score)
    
    return final_scores