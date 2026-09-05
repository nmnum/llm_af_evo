def modifier(context):
    """Reward candidates that explore under-covered regions of objective space based on GP predictive variance across objectives."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute normalized sum of variances (1 - std^2 / range^2) for each candidate
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        variance_norm_sum = 0.0
        total_variance = 0.0
        
        # For each objective, compute normalized predictive variance (1 - std²/range²)
        for name in names:
            mean_val = gp[name]["mean"] 
            std_val = gp[name]["std"]
            
            range_val = front_range[name]
            if abs(range_val) < 1e-8: continue
            
            # Normalize standard deviation to [0, 1] and compute variance
            normalized_std_sq = (std_val / range_val)**2  
            var_norm = max(0.0, 1 - normalized_std_sq)
            
            total_variance += std_val**2 
            variance_norm_sum += var_norm
            
        # Scale bonus by the inverse of mean uncertainty across objectives to encourage exploring low-uncertainty regions
        if len(names) > 0 and sum(gp[name]["std"] for name in names) > 1e-8:
            avg_std = total_variance / float(len(names))
            
            values.append(variance_norm_sum * (avg_std + 1.0)) 
        else:  
            # fallback to baseline
            values.append(0)
    
    return values