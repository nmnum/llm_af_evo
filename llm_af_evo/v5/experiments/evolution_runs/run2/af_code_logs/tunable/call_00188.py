def score_pool(context):
    """Score candidates by normalized hypervolume improvement adjusted with inverse squared distance to latest Pareto front centroid."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    pf = context["pareto_front"]
    
    # Compute the centroid of current pareto front
    if len(pf) == 0:
        front_centroid = ref_point - 1.0  # fallback to avoid division by zero or NaN issues  
    else: 
        front_centroid = np.mean(pf, axis=0)
        
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        acq_norm = cand["acq_value_norm"] 
        
        # Predicted mean vector
        pred_mean = np.array([gp[name]["mean"] for name in names])
          
        # Distance from candidate to front centroid  
        dist_to_front_centroid = np.linalg.norm(pred_mean - front_centroid)
        
        # Inverse squared distance as novelty bonus (higher when farther) 
        if dist_to_front_centroid == 0:
            novel_bonus = float('inf')  # avoid division by zero
        else:    
            novel_bonus = 1.0 / (dist_to_front_centroid ** 2)
            
        scores.append(acq_norm + 0.5 * novel_bonus) 
        
    return scores