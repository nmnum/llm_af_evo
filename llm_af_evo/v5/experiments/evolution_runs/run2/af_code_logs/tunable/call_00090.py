def score_pool(context):
    """Estimates hypervolume improvement by resampling Pareto fronts and measuring candidate's contribution to each."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Prepare reference point for HV calculation  
    ref_point = context['ref_point']
    
    n_resamples = 20
    scores = []
        
    Y_obs = context['Y_obs'] 
      
    def dominates(p, q):
        return (np.all(q >= p) and np.any(q > p))
            
    # For each candidate in the pool  
    for cand in context["pool"]:
        improvement_sum = 0.0
          
        # Bootstrap resamples of observations    
        for _ in range(n_resamples): 
            inds = np.random.choice(len(Y_obs), size=len(Y_obs), replace=True)
            Y_sampled = Y_obs[inds]
            
            # Compute non-dominated points (Pareto front) from this sample  
            pf_mask = np.ones(len(Y_sampled), dtype=bool)
          
            for i in range(len(Y_sampled)):
                if not pf_mask[i]: continue
                p_i = Y_sampled[i] 
                
                for j in range(i+1, len(Y_sampled)):   
                    if not pf_mask[j]: continue  
                    
                    # Check dominance relation between points
                    q_j = Y_sampled[j]
                  
                    if dominates(p_i, q_j):
                        pf_mask[j] = False  # Remove dominated point
                        
            sampled_pf = Y_sampled[pf_mask]

            
            cand_obj = np.array([cand["gp_posterior"][name]["mean"] for name in names])
              
            # Calculate hypervolume of PF with candidate added  
            if len(sampled_pf) == 0:
                hv_with_cand = np.prod(ref_point - cand_obj)
            else: 
                pf_plus_candidate = np.vstack([sampled_pf, cand_obj])   
                
                # Use simple but correct vectorized HV calc
                hypervolumes = []
              
                for point in pf_plus_candidate:
                    if all(point >= ref_point): continue  # Point outside domain  
                    
                    vol = np.prod(ref_point - point)
                    hypervolumes.append(vol) 
                  
                hv_with_cand = sum(hypervolumes)

            # Calculate HV of PF alone (without candidate)
            if len(sampled_pf) == 0:
                hv_without_cand = 0.0
            else:  
                
                hypervolumes_orig = []
              
                for point in sampled_pf:
                    vol = np.prod(ref_point - point)
                    hypervolumes_orig.append(vol)

                hv_without_cand = sum(hypervolumes_orig) 
                 
            
            improvement_sum += (hv_with_cand - hv_without_cand)
                
        # Average over all resamples
        avg_improvement = improvement_sum / n_resamples  
        
        scores.append(avg_improvement)
         
    return scores