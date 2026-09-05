def score_pool(context):
    """Estimates hypervolume improvement by bootstrapping Pareto fronts and computing candidate impact on each."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Bootstrap resampling of Y_obs to estimate front uncertainty
    n_resamples = 15
    ref_point = context['ref_point']
    y_obs = context['Y_obs'] 
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Predicted objective values (already oriented to be maximized)
        pred_obj_vals = np.array([gp_posterior[name]["mean"] for name in names])
        
        total_improvement = 0.0
        n_fronts_considered = 0
        
        # Perform bootstrap resampling of observations and compute hypervolume improvement per sample front 
        for _ in range(n_resamples):
            boot_indices = np.random.choice(len(y_obs), size=len(y_obs), replace=True)
            y_boot = y_obs[boot_indices]
            
            if len(np.unique(y_boot, axis=0)) < 2:
                continue
            
            # Compute non-dominated front from bootstrap sample
            dominated_mask = np.zeros(len(y_boot), dtype=bool) 
            for i in range(len(y_boot)):
                dom_by_any = False  
                for j in range(len(y_boot)):                    
                    if all(y_boot[j] >= y_boot[i]) and any(y_boot[j] > y_boot[i]):
                        dom_by_any = True
                        break
                        
                dominated_mask[i] = dom_by_any
                
            front_points = y_boot[~dominated_mask]
            
            # Compute hypervolume improvement of adding the candidate to this bootstrap front 
            if len(front_points) == 0:
                hv_with_cand = np.prod(ref_point - pred_obj_vals)
                hv_without_cand = 0.0
            else:  
                combined_fronts = np.vstack([front_points, pred_obj_vals])
                
                # Compute hypervolume of front with candidate 
                volumes_with = []
                for pt in combined_fronts:
                    vol_contrib = np.prod(np.maximum(ref_point - pt, 0))
                    if not any(all(combined_fronts[i] >= pt) and any(combined_fronts[i] > pt)
                               for i in range(len(combined_fronts)) if (combined_fronts[i]==pt).all()):
                        volumes_with.append(vol_contrib)

                hv_with_cand = sum(volumes_with)


                 # Compute hypervolume of front without candidate  
                volumes_without = []
                for pt in front_points:
                    vol_contrib = np.prod(np.maximum(ref_point - pt, 0))
                    if not any(all(front_points[i] >= pt) and any(front_points[i] > pt)
                               for i in range(len(front_points)) if (front_points[i]==pt).all()):
                        volumes_without.append(vol_contrib)

                hv_without_cand = sum(volumes_without)


            improvement = max(0.0, hv_with_cand - hv_without_cand) 
            total_improvement += improvement
            n_fronts_considered += 1
            
        # Average across resamples (if any)
        if n_fronts_considered > 0:
            avg_improvement = total_improvement / float(n_fronts_considered)
        else:  
            avg_improvement = 0.0
          
        scores.append(avg_improvement)

    return scores