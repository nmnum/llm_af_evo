def score_pool(context):
    """Blend acquisition value with a coverage-gap term: candidates near sparse front regions get higher scores."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Use Y_obs if pareto_front is too small for k=3 nearest neighbors
    pf = context['pareto_front']
    obs = context['Y_obs'] 
    use_pf = len(pf) >= 3
    ref_points = pf if use_pf else obs
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Get candidate's predicted objectives (already maximized)
        pred_obj = np.array([gp[name]["mean"] for name in names])
        
        # Compute distances to k=3 nearest front points
        dists = []
        if len(ref_points) > 0:
            diff = ref_points - pred_obj 
            sq_dists = np.sum(diff**2, axis=1)
            
            sorted_indices = np.argsort(sq_dists)[:min(3, len(ref_points))]
            for idx in sorted_indices:  
                dists.append(np.sqrt(sq_dists[idx]))
        
        # Mean of k nearest distances (or 0 if no reference points) 
        mean_dist =np.mean(dists) if dists else 0.0
        
        # Blend with acquisition value
        acq_value_norm = cand["acq_value_norm"]
        score = acq_value_norm + 0.1 * mean_dist  
        
        scores.append(score)
    
    return scores