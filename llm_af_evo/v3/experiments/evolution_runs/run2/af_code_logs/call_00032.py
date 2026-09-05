def score_pool(context):
    """Progress-aware hypervolume gain estimate blended with novelty; avoids uncertainty weighting and diversity heuristics."""
    X_obs = context["X_obs"]
    names = context["objective_names"] 
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    progress = context["campaign"]["progress"]
    
    # Estimate hypervolume contribution of each candidate
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        means = [gp[name]["mean"] for name in names]
        
        # Normalize the predicted point relative to ref_point and front range  
        normalized_means = [(ref_point[i] - m) / front_range[names[i]] for i, m in enumerate(means)]
        
        # Compute hypervolume contribution as product of distances from reference
        hv_contribution = np.prod(np.maximum(0.0, normalized_means))
        
        # Add novelty bonus (inverse distance to nearest observed point)
        cand_x = cand["x"]
        if len(X_obs) > 0:
            min_dist = float(np.linalg.norm(X_obs - cand_x, axis=1).min())
            novelty_bonus = np.exp(-min_dist * 5.0) # Scale down the bonus
        else: 
            novelty_bonus = 1.0
            
        scores.append(hv_contribution + (novelty_bonus * max(0.2, 1 - progress)))
    
    return scores