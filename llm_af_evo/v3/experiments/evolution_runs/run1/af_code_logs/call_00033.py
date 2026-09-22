def score_pool(context):
    """Estimates improvement potential by resampling candidates' posteriors and scoring based on hypervolume expansion probability."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    pareto_front = context["pareto_front"] 
    X_obs = context["X_obs"]
    
    # Sample from each candidate's posterior to estimate HV improvement
    n_samples = 50
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Generate samples from the joint GP distribution of objectives 
        means = [gp[name]["mean"] for name in names]
        stds = [gp[name]["std"] for name in names] 
        
        # Simple diagonal covariance matrix (assumes independence)
        cov_matrix = np.diag([s**2 for s in stds])
        
        samples = np.random.multivariate_normal(means, cov_matrix, n_samples) 
        samples = np.maximum(samples, ref_point)  # Ensure no points fall below reference
        
        # Compute hypervolume improvement estimate
        hv_improvement = []
        for sample in samples:
            # Estimate HV increase by comparing to current Pareto front  
            expanded_front = np.vstack([pareto_front, sample])
            
            # Simple approximate HV (normalized) 
            if len(expanded_front) == 1: 
                hypervolume = max(0.0, ref_point[0] - sample[0]) * max(0.0, ref_point[1] - sample[1])
            else:
                # Use a simple box-based approximation
                mins = np.min(expanded_front, axis=0)
                maxs = np.max(expanded_front, axis=0) 
                
                hypervolume = 1.0
                for i in range(len(names)):
                    if sample[i] < ref_point[i]:
                        # Only count contribution to HV when point is within bounds  
                        hv_contribution = (ref_point[i] - mins[i]) * max(0, sample[1-i] - min(ref_point[1-i], mins[1-i]))
                        hypervolume *= hv_contribution
                    else:
                        hypervolume = 0.0
                        
            # Normalize by reference point range to avoid scale issues  
            norm_factor = np.prod([ref_point[i] for i in range(len(names))])
            
            if norm_factor > 0: 
                normalized_hv_improvement = hv_conprovement / norm_factor
            else:
                normalized_hv_improvement = 0.0
                
            hv_improvement.append(normalized_hv_improvement)
        
        # Use mean HV improvement as score  
        scores.append(np.mean(hv_improvement))
    
    return scores