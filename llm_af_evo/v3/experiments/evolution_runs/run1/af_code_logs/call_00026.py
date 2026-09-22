def score_pool(context):
    """Estimates improvement potential by resampling observed data under noise to assess candidate dominance probability."""
    n_samples = 30
    ref_point = context["ref_point"]
    
    # Resample Y_obs with added Gaussian noise matching GP stds for each objective 
    y_noisy = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        
        sample_y = np.array([
            [np.random.normal(
                context['Y_obs'][i][j],  
                gp_posterior[context["objective_names"][j]]["std"]) 
             for j in range(len(context["objective_names"]))] 
            for _ in range(n_samples)
        ])
    
        y_noisy.append(sample_y)

    scores = []
    for i, cand in enumerate(context["pool"]):
        
        # Compute hypervolume improvement estimates from noisy samples
        sample_y = y_noisy[i]
        if len(context['pareto_front']) == 0:
            hv_values = np.prod(np.maximum(sample_y - ref_point, 0), axis=1)
            
        else: 
            # For each sampled point compute its contribution to HV wrt current front  
           hv_values = []
            for s in sample_y:

                dominated = False
                for q in context['pareto_front']:
                    if np.all(q >= s) and np.any(q > s):
                        dominated = True
                        
                volume = np.prod(np.maximum(s - ref_point, 0))
                
                # Discount contribution of dominated samples (they don't improve the front)
                discounted_volume = volume * 0.1 if dominated else volume
                hv_values.append(discounted_volume)

            hv_values = np.array(hv_values) 
        
        mean_hv_imp = np.mean(hv_values)
        
        scores.append(mean_hv_imp)
    
    return scores