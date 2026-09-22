def score_pool(context):
    """Resample noisy observations to assess candidate robustness and hypervolume expansion potential."""
    import numpy as np
    
    # Extract relevant data
    names = context["objective_names"]
    X_obs = context["X_obs"] 
    Y_obs = context["Y_obs"]
    pareto_front = context["pareto_front"]
    
    scores = []
    for cand in context["pool"]:
        acq_value = cand["acq_value_norm"]
        
        # Estimate robustness by resampling noisy observations
        n_resamples = 10
        
        # Simulate noise around observed points (simple Gaussian)
        y_noisy_samples = Y_obs + np.random.normal(0, 0.05 * pareto_front.max(axis=0), size=Y_obs.shape)

        # Estimate how often this candidate would dominate or be dominated
        cand_pred_means = [cand["gp_posterior"][name]["mean"] for name in names]
        
        dom_count = 0.
        non_dom_count = 0.

        # For each resampled observation, check dominance relationship with current candidate  
        for y_noisy_row in y_noisy_samples:
            if all(y_c < yc for (y_c,yc) in zip(y_noisy_row,cand_pred_means)):
                dom_count +=1
            elif not any(yc <= y_c for (y_c,yc) in zip(y_noisy_row,cand_pred_means)): 
                non_dom_count += 1

        # Robustness score: how frequently candidate is dominated or dominates others  
        robust_score = max(dom_count / n_resamples - non_dom_count/n_resamples ,0)
        
        scores.append(acq_value + 0.3 * robust_score)

    return scores