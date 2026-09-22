def score_pool(context):
    """Estimates improvement potential by resampling candidates' posteriors and scoring based on how often they dominate current front points."""
    import numpy as np
    
    # Use Monte Carlo sampling from each candidate's GP posterior 
    n_samples = 20  
    names = context["objective_names"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        
        # Sample objectives many times from this candidate’s predictive distribution
        samples = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"]
            std_val = gp[name]["std"]
            
            if std_val > 0:
                samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
            else: 
                # If no uncertainty (very rare), just use the point estimate
                samples[:, i] = mean_val
        
        # Count how many times this sample dominates any current front point  
        dom_count = 0.0

        for s in samples:
            is_dominated_by_front = False
            
            # Check if this sampled objective vector is dominated by a known non-dominated solution 
            for pf_point in context["pareto_front"]:
                if all(s[i] >= pf_point[i] and not np.isclose(s[i],pf_point[i]) for i in range(len(names))):
                    is_dominated_by_front = True
                    break
                    
            # If it's NOT dominated by any front point, then the candidate has potential to improve hypervolume  
            
            if not is_dominated_by_front:
                dom_count += 1.0

        score = dom_count / n_samples
        
        scores.append(score)
        
    return scores