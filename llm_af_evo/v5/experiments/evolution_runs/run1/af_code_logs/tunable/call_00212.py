def score_pool(context):
    """Score candidates based on how much they extend under-covered regions of the Pareto front by adding a coverage-gap term to the acquisition value."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Use Y_obs if pareto_front is too small for k=3
    pf = context['pareto_front']
    obs = context['Y_obs'] 
    use_pf = len(pf) >= 3
    reference_points = pf if use_pf else obs
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted objectives (already maximized)
        pred_obj = np.array([gp[name]["mean"] for name in names])
        
        # Compute distances to k nearest points
        if len(reference_points) == 0:
            coverage_gap_score = 0.0  
        else: 
            dists = [np.linalg.norm(pred_obj - ref_point, ord=2)
                     for ref_point in reference_points]
            sorted_dists = np.sort(dists)[:3] # k nearest
            mean_dist = float(np.mean(sorted_dists)) if len(sorted_dists) > 0 else 0.0
            
        coverage_gap_score = mean_dist
        
        blended_score = context["pool"][0]["acq_value_norm"] + (coverage_gap_score * 1e-3)
        
        scores.append(blended_score)

    return scores