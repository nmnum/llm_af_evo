def score_pool(context):
    """Score candidates by expected hypervolume improvement adjusted for posterior uncertainty and Pareto dominance risk."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context['ref_point']
    front = context['pareto_front'] 
    n_obs = len(context['X_obs'])
    
    # Normalize acquisition value by observed history size
    acq_norm_factor = 1.0 / (n_obs + 1)
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted mean objectives 
        means = np.array([gp[name]["mean"] for name in names])
        
        # Estimated hypervolume improvement from this candidate
        hv_improvement = 1.0
        
        if len(front) > 0:
            front_plus_candidate = np.vstack((front, means))
            
            # Compute reference point adjusted to include the new candidate  
            ref_adj = np.maximum(ref_point, means)
            
            # Simple hypervolume estimate: volume between current front and expanded version
            hv_original = np.prod(np.maximum(front.max(axis=0) - ref_point, 0)) if len(front) > 0 else 1.0
            
            try:
               hv_with_candidate = np.prod(ref_adj - means)
                
                # Discount improvement for candidates that are likely to be dominated  
                dominates_count = sum(1 for f in front 
                                    if all(f[i] >= means[i] and f[i] != means[i]) for i in range(len(names)))
            
                dominance_risk_factor = max((0.5 - 0.2 * min(dominates_count, 3)), 0.1)
                
            except:
                hv_with_candidate = np.prod(np.maximum(ref_adj - ref_point, 0))
                dominates_count = len(front)  
                dominance_risk_factor = 0.9
                
        else: 
            # No front yet; just compute raw volume
            hypervolume_raw = max(1e-8, np.prod(means - ref_point)) if all(m > r for m,r in zip(means,ref_point)) else 1.
            
        
        hv_improvement *= (hv_with_candidate / (max(hv_original, 1.0) + 1e-6))
        # Add uncertainty penalty
        sigma_sum = sum(gp[name]["std"]**2 for name in names)
        score = hv_improvement * np.exp(-sigma_sum/4.) 
          
        scores.append(score)

    return scores