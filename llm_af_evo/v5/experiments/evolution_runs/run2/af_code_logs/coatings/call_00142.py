def score_pool(context):
    """Blend acquisition value with a dynamic diversity reward that emphasizes spreading out across objective space."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute distances from each candidate to all previously observed points in obj-space
    y_obs = context["Y_obs"] 
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Get predicted means and normalize them using the front range (for consistent scaling)
        pred_means_normed = np.array([gp[name]["mean"] / front_range[name] for name in names])
    
        min_dist_to_observed = float('inf')
        for y in y_obs:
            dist = np.linalg.norm(pred_means_normed - y)  # L2 distance
            if dist < min_dist_to_observed:
                min_dist_to_observed = dist

        acq_score = cand["acq_value_norm"]
        
        # Normalize the minimum distance to observed points, then apply a non-linear boost for diversity  
        normalized_diversity = np.exp(-min_dist_to_observed)  # exponential decay
          
        score = acq_score + 0.2 * normalized_diversity   # small weight on novelty
        
        scores.append(score)
    
    return scores