def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with an uncertainty-adjusted progress signal that rewards candidates whose predicted objectives lie in under-explored regions of objective space, measured by how far they are from existing front points relative to their own predictive standard deviation."""
    names = context["objective_names"]
    ref_point = context["ref_point"]
    pareto_front = context["pareto_front"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Compute normalized distance from candidate to nearest front point
        if len(pareto_front) == 0:
            dist_to_front_norm = float('inf')
        else:
            pred_obj = np.array([gp[name]["mean"] for name in names])
            distances = [np.linalg.norm(pred_obj - fp, ord=2) / 
                         (1e-9 + np.sqrt(sum(gp[n]['std']**2 for n in names)))  
                         for fp in pareto_front]
            dist_to_front_norm = min(distances)
        
        # Use acquisition value as the main score component
        base_score = cand["acq_value_norm"]
        
        # Adjust by uncertainty-normalized front distance: higher is better (more unexplored, more promising region)  
        if np.isinf(dist_to_front_norm):
            adjusted_uncertainty_signal = 0.5 
        else:
            adjusted_uncertainty_signal = max(1e-6, dist_to_front_norm)
            
        # Combine acquisition value and the uncertainty-adjusted progress signal
        score = base_score + (adjusted_uncertainty_signal * 0.2)  
        
        scores.append(score)

    return scores