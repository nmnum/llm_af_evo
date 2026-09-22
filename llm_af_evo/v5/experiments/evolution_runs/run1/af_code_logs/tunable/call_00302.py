def score_pool(context):
    """Reward candidates near sparse regions of the Pareto front using k-nearest neighbor distances to the front."""
    names = context["objective_names"]
    
    # Use Y_obs if pareto_front is too small for meaningful density estimation
    pf = context['pareto_front']
    if len(pf) < 3:
        reference_points = context['Y_obs'] 
    else:
        reference_points = pf
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Get candidate's predicted objectives
        pred_obj = np.array([gp[name]["mean"] for name in names])
        
        if len(reference_points) == 0:
            coverage_score = 0.0 
        else:
            distances = [np.linalg.norm(pred_obj - ref_point, ord=2)
                         for ref_point in reference_points]
            
            # Get mean of k nearest (k=3 or less than total points available)
            sorted_distances = np.sort(distances)[:min(3, len(reference_points))]
            coverage_score = np.mean(sorted_distances) if len(sorted_distances) > 0 else 0.0
            
        blended_score = context["pool"][0]["acq_value_norm"] + (coverage_score * 1e-4)
        
        scores.append(blended_score)

    return scores