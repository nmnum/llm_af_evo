def score_pool(context):
    """Estimates improvement potential by resampling noisy observations to infer dominance changes."""
    import numpy as np
    
    scores = []
    n_obs = len(context["X_obs"])
    
    if n_obs < 1:
        # If no observations, fall back to acquisition value only (exploitation)
        for cand in context["pool"]:
            scores.append(cand['acq_value_norm'])
        return scores

    ref_point = context["ref_point"]
    pareto_front = context["pareto_front"]

    # Compute hypervolume improvement estimates via resampling
    n_samples = 20   # number of noisy samples to draw per candidate
    
    for cand in context["pool"]:
        
        gp_posterior = cand['gp_posterior']
        f1_mean, f1_std = gp_posterior['f1']['mean'], gp_posterior['f1']['std'] 
        f2_mean, f2_std = gp_posterior['f2']['mean'], gp_posterior['f2']['std']

        # Sample noisy predictions for this candidate
        samples_f1 = np.random.normal(f1_mean, f1_std, n_samples)
        samples_f2 = np.random.normal(f2_mean, f2_std, n_samples)

        # Compute hypervolume improvement by seeing how many of the resampled points dominate or are better than current front  
        
        hv_improvements = []
 
        for i in range(n_samples):
            point_i = [samples_f1[i], samples_f2[i]]
            
            if len(pareto_front) == 0:
                # No existing Pareto front, so any new point improves hypervolume
                hp_vol = (ref_point[0] - point_i[0]) * (ref_point[1] - point_i[1])
                
            else: 
                dominates_any_existing = False
                
                for pf in pareto_front:
                    if ((point_i[0] >= pf[0]) and (point_i[1] > pf[1])) or \
                       ((point_i[0] > pf[0])  and (point_i[1] >= pf[1])):
                        dominates_any_existing = True
                        break
                        
                # If this point doesn't dominate any existing front points, compute hypervolume contribution 
                
                if not dominates_any_existing:
                    hp_vol = min(ref_point[0], point_i[0]) * min(ref_point[1], point_i[1])
                    
                else:  # dominated by current PF
                     hp_vol = - (ref_point[0] + ref_point[1])

            hv_improvements.append(hp_vol)

        expected_hv_imp = np.mean(np.array(hv_improvements))
        
        scores.append(expected_hv_imp)
    
    return scores