def score_pool(context):
    """Resample posterior objectives under noise to assess domination risk and reward candidates that avoid it."""
    
    names = context["objective_names"]
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Use acquisition value as baseline
    scores = list(acq_values)
    
    n_samples_per_candidate = 50
    
    X_obs = context["X_obs"]
    Y_obs = context["Y_obs"]
    ref_point = context["ref_point"]
    front_range = context["pareto_front_range"] 
     # Normalize reference point for domination checks
    norm_ref = np.array([context['ref_point_by_name'][name] for name in names])
    
    if len(X_obs) > 0:
        # Compute how many times each candidate would be dominated by sampled points from history  
        
        dominance_counts = [0.0]*len(context["pool"])
                
        for _ in range(n_samples_per_candidate):
            # Sample noisy observations
            sample_Ys = []
            
            for i, y_obs in enumerate(Y_obs): 
                noisy_y = []   
                    
                for j, obj_val in enumerate(y_obs):
                    mu, sigma = Y_obs[i][j], 1e-6   # Very small noise to simulate sampling
                
                    if sigma <= 0:
                        sampled = mu
                    else:  
                        sampled = np.random.normal(mu, sigma)
                        
                    noisy_y.append(sampled) 
                    
                sample_Ys.append(noisy_y)

            for cand_idx, cand in enumerate(context["pool"]):
                
                gp_posterior = cand['gp_posterior']
                  
                # Sample candidate's objectives
                samp_obj = []
                for name in names:
                    mu, sigma = gp_posterior[name]["mean"], gp_posterior[name]["std"]
                    
                    if sigma <= 0: 
                        sampled_val = mu  
                    else:
                        sampled_val = np.random.normal(mu, sigma)
                        
                    samp_obj.append(sampled_val)

                # Check how many historical points dominate this candidate's sample
                n_dominated_by_history = sum(1 for y in sample_Ys if all(y[i] >= (samp_obj[i]- 1e-8) 
                                                                          for i in range(len(names))))

                dominance_counts[cand_idx] += float(n_dominated_by_history)

        # Lower score for candidates that are often dominated by sampled history
        avg_domination = np.array(dominance_counts)/n_samples_per_candidate
        
        scores = [s - 0.5 * d for s, d in zip(scores, avg_domination)]
        
    return scores