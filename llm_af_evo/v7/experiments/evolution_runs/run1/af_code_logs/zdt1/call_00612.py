def modifier(context):
    """Noise-resampled dominance bonus: estimate how often a candidate would dominate under posterior noise samples."""
    import numpy as np
    
    names = context["objective_names"]
    y_obs = context["Y_obs"] 
    x_obs = context["X_obs"]
    
    # Compute the number of observations
    n_obs = len(y_obs)
    
    if n_obs == 0:
        return [0.0] * len(context["pool"])
        
    values = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        x_cand = cand["x"]
                
        # Sample noise from GP posteriors
        n_samples = 50
        
        samples_y = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"] 
            samples_y[:,i] = np.random.normal(mean_val, std_val, n_samples)
            
        # Check how often the candidate dominates observed points
        dom_count = 0
        
        for i in range(n_obs):
            y_i = y_obs[i]
                        
            if all(samples_y[j,:] >= y_i) and any(samples_y[j,:] > y_i) :
                dom_count +=1
                
        
        bonus_score = (dom_count / n_samples)
                
        values.append(bonus_score * 0.2)

    return values