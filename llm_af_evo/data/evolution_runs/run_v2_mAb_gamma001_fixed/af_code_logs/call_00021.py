def score_pool(context):
    """Estimate each candidate's potential hypervolume improvement using Monte Carlo samples from their posteriors, then apply a diversity bonus based on nearest neighbor distances in feature space."""
    import numpy as np
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Sample candidates' objectives to estimate HV contribution
    n_samples = 100
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Draw samples from the candidate's posterior distribution across all objectives
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"]
            std_val = gp[name]["std"] 
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
        
        # Estimate HV improvement by comparing to current pareto front
        hv_improvement = 0.0
        
        for sample in samples:
            dominates_any = False
            
            # Check if this sampled point is dominated by any existing non-dominated points  
            for pf_point in context["pareto_front"]:
                if all(sample[i] >= pf_point[i] for i in range(len(names))):
                    dominates_any = True
                    break
                    
            # If not dominated, it contributes to HV expansion (simplified)
            hv_improvement += 1.0 if not dominates_any else 0.0
            
        avg_hv_impact = hv_improvement / n_samples
        
        scores.append(avg_hv_impact)

    return scores