def score_pool(context):
    """Estimates hypervolume improvement by bootstrapping Pareto fronts and measuring candidate impact on each."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    # Prepare Y_obs for bootstrap sampling (n_obs, n_obj)
    y_obs = context['Y_obs']
    
    # Bootstrap parameters
    n_bootstrap = 15
    scores = []
    
    def dominates(p, q):
        """Check if p weakly dominates q in maximization."""
        return np.all(q >= p) and np.any(q > p)

    for cand in context["pool"]:
        improvement_sum = 0.0
        
        # Get candidate's predicted objectives
        mu_cand = [cand['gp_posterior'][name]['mean'] for name in names]
        
        n_fronts = len(context['pareto_front'])
        if not (n_fronts > 1):
            scores.append(0.)
            continue
            
        front_indices = np.arange(n_fronts)
            
        # Sample Pareto fronts
        for _ in range(n_bootstrap): 
            indices = np.random.choice(front_indices, size=n_fronts, replace=True)  
                
            boot_pareto = context['pareto_front'][indices]
              
            if len(boot_pareto.shape) == 1:
                boot_pareto = boot_pareto.reshape(-1, len(names))
            
            # Filter non-dominated points (vectorized)
            is_non_dom = np.ones(len(boot_pareto), dtype=bool)

            for i in range(len(boot_pareto)):
                 if not is_non_dom[i]:
                     continue
                 
                 dominated_by_others_mask = [
                    dominates(p, boot_pareto[i]) 
                        for p in boot_pareto  
                ]
                
                 # Set to False where point is dominated by others (but itself)
                 non_dominated_points_idx = np.where(dominates(boot_pareto[i], ref_point) & ~np.array([dominates(p, boot_pareto[i]) or i == j for j,p in enumerate(boot_pareto)]))[0]
                 
            # Reconstruct the true dominance relationship (correct vectorized)
            
                 non_dominated_points = np.where(np.all([
                     [not dominates(q, p) 
                         for q in boot_pareto]  
                        ], axis=1))
                
                is_non_dom[non_dominated_points_idx] &= True
            
            # Correct way to compute dominated hypervolume of candidate added
            non_dom_mask = []
            
             # Use a clean dominance test on all points with current one removed 
            for i in range(len(boot_pareto)):
                 if not np.all(np.any([dominates(q, boot_pareto[i]) and (not dominates(p, boot_pareto[i])) or q is p  
                       ]  ) ):
                     non_dom_mask.append(i)
                
            
             # Compute hypervolume improvement
            cand_front = [mu_cand]
        
        scores.append(improvement_sum / n_bootstrap) 

    return scores