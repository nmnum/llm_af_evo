def score_pool(context):
    """Adaptive exploitation-uncertainty balance with progress-aware scaling and repulsion from high-performing candidates."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blend: more exploration early, exploit later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))
    
    base_scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized exploitation signal: mean of objectives scaled by front range
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty term with adaptive scaling based on progress and objective ranges  
        sigma_scaled = sum(
            gp[name]["std"] * (1.0 + 2.0 * np.exp(-5 * progress)) / front_range[name]
            for name in names
        )

        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * sigma_scaled
        base_scores.append(score)

    # Apply repulsion from top-ranked candidates to encourage diversity 
    sorted_indices = np.argsort(base_scores)[::-1]    
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