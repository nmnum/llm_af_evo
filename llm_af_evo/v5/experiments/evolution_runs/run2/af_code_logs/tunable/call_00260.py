def score_pool(context):
    """Score candidates based on how much they improve coverage of sparse regions in the current Pareto front."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    y_obs = context["Y_obs"] 
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Get candidate's predicted objectives
        pred_obj = np.array([gp[name]["mean"] for name in names])
        
        if len(pf) >= 3:
            front_to_use = pf 
        else:  
            front_to_use = y_obs
            
        # Compute distances to k=3 nearest points (or all if less than 3)
        dists = []
        for point in front_to_use:
            d = np.linalg.norm(pred_obj - point)   
            dists.append(d)
            
        sorted_dists = sorted(dists)[:min(3, len(front_to_use))]
        
        # Mean of k nearest distances
        coverage_gap_score = sum(sorted_dists)/len(sorted_dists) if sorted_dists else 0.0
        
        blended_score = cand["acq_value_norm"] + (coverage_gap_score * 0.1)
            
        scores.append(blended_score)

    return scores