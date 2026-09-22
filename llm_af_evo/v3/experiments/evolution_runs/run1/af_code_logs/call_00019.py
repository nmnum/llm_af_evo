def score_pool(context):
    """Estimates improvement potential by resampling observed data under noise to assess robustness of predictions."""
    n_samples = 20
    ref_point = context["ref_point"]
    
    # Resample Y_obs with added Gaussian noise matching GP stds for each objective 
    y_noisy = []
    for i, obs in enumerate(context["Y_obs"]):
        noisy_y = np.random.normal(obs, [context["pool"][i]["gp_posterior"]["f1"]["std"], context["pool"][i]["gp_posterior"]["f2"]["std"]])
        y_noisy.append(noisy_y)
    
    # Compute hypervolume for each resampled front
    hv_scores = []
    for _ in range(n_samples):
        sampled_fronts = [y_noisy[i] for i in np.random.choice(len(y_noisy), size=len(y_noisy), replace=True)]
        
        if not sampled_fronts:
            hv_scores.append(0.0)
            continue
            
        # Compute non-dominated points from resampled front
        pf_samples = []
        for y in sampled_fronts: 
            is_dominated = False  
            for other_y in sampled_fronts:
                if np.all(other_y >= y) and any(other_y > y):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pf_samples.append(y)
        
        # Compute hypervolume of the resampled front relative to ref_point  
        hv_sample = 0.0 
        for p in pf_samples:    
            vol = np.prod(np.maximum(p - ref_point, 0))
            hv_sample += vol
            
        hv_scores.append(hv_sample)

    mean_hv_imp = np.mean(hv_scores)
    std_hv_imp = np.std(hv_scores) 
    
    # Score candidates by how much their prediction would increase the expected HV
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        pred_mean_f1, pred_std_f1 = gp_posterior["f1"]["mean"], gp_posterior["f1"]["std"] 
        pred_mean_f2, pred_std_f2 = gp_posterior["f2"]["mean"], gp_posterior["f2"]["std"]

        # Sample from candidate's GP posterior
        samples_cand = np.array([
            [np.random.normal(pred_mean_f1, pred_std_f1),  
             np.random.normal(pred_mean_f2, pred_std_f2)]
            for _ in range(n_samples)
         ])
        
        hv_improvement_per_sample = []
        # For each sample from candidate's GP posterior
        for s_cand in samples_cand:
            
            temp_fronts = [y_noisy[i] if i < len(y_noisy) else None 
                           for i in np.random.choice(len(y_noisy), size=len(y_noisy)-1, replace=True)]
                
            # Add candidate to front temporarily
            extended_front = list(temp_fronts)
            extended_front.append(s_cand)

            pf_temp = []
            dominated_flags = [False] * len(extended_front) 
            
            for i in range(len(extended_front)):
                if not dominated_flags[i]:
                    is_dominated_by_others = False 
                    
                    # Check dominance against all others
                    j_start_idx = 0  
                    while (j_start_idx < len(extended_front)) and not is_dominated_by_others:
                        if i != j_start_idx:    
                            y_i, y_j = extended_front[i], extended_front[j_start_idx]
                            
                            dominated_or_equal_check_passed = True
                            for dim in range(len(y_i)):
                                # If any dimension of candidate > other's (strictly), not dominated  
                                if s_cand[dim] >= extended_front[j_start_idx][dim]:
                                    continue 
                                
                                # Else, it was <= and may be dominated by this one   
                                else:
                                    dominated_or_equal_check_passed = False
                                    
                            if dominated_or_equal_check_passed:  # All dims of candidate are not strictly > other's  
                                                    
                                is_dominated_by_others = True 

                        j_start_idx +=1
                        
                    if not is_dominated_by_others and s_cand[0] >= ref_point[0]:
                        pf_temp.append(s_cand)
                        
            hv_imp_sampled = 0. 
            for p in pf_temp:
                vol = np.prod(np.maximum(p - ref_point, 0))
                hv_imp_sampled += vol
                
            # Compute hypervolume difference with and without the candidate
            if len(pf_temp) == 1:  
               diff_hv = max(0., (hv_imp_sampled)) 
            else:
                try:
                    front_without_cand= [y for i, y in enumerate(temp_fronts)]
                    
                    # Get PF from this reduced set too   
                    pf_reduced_set=[]
                    

        scores.append