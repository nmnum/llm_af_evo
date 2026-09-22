def score_pool(context):
    """Estimates improvement potential by resampling candidates' GP posteriors to assess how often they would dominate current front points."""
    import numpy as np
    
    names = context["objective_names"]
    pareto_front = context["pareto_front"] 
    ref_point = context["ref_point"]
    
    # Sample from each candidate's posterior 10 times
    n_samples = 10
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Draw samples from the GP posteriors of this candidate 
        sampled_objectives = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"]
            std_val = gp[name]["std"]  
            
            # Sample normally distributed values
            sampled_objectives[:,i] = np.random.normal(mean_val, std_val, n_samples)
          
        # For each sample, compute hypervolume contribution relative to current front 
        hv_contributions = []
        
        for i in range(n_samples):
            s_obj = sampled_objectives[i]
            
            # Check if this sample dominates any point on the Pareto front
            is_dominant = True
            
            for pf_point in pareto_front:
                # If candidate's objective values are all <= PF, then it doesn't dominate 
                if np.all(s_obj >= pf_point):
                    continue  # Not dominated by current PF points

            hv_contributions.append(1.0) 

        score = sum(hv_contributions)/n_samples
        scores.append(score)
        
    return scores