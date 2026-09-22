def score_pool(context):
    """Estimates hypervolume improvement by bootstrapping Pareto fronts and measuring candidate contribution to each."""
    import numpy as np
    
    # Bootstrap settings
    n_draws = 20  
    names = context["objective_names"]
    
    scores = []
    for cand in context["pool"]:
        x_cand = cand["x"] 
        y_cand_pred = [cand["gp_posterior"][name]["mean"] for name in names]
        
        total_improvement = 0.0
        n_valid_draws = 0
        
        # Bootstrap resampling of Y_obs  
        for _ in range(n_draws):
            indices = np.random.choice(len(context['Y_obs']), size=len(context['Y_obs']), replace=True)
            y_bootstrapped = context["Y_obs"][indices]
            
            if len(y_bootstrapped) == 0:
                continue
                
            # Compute non-dominated set for this bootstrap sample
            nondom_mask = np.ones(len(y_bootstrapped), dtype=bool)

            for i in range(len(y_bootstrapped)):
                y_i = y_bootstrapped[i] 
                
                if not nondom_mask[i]:
                    continue
                    
                dominates_other = (y_bootstrapped >= y_i).all(axis=1) & \
                                  ((y_bootstrapped > y_i).any(axis=1))
                  
                # Mark dominated points as non-nondominated
                nondom_mask[dominates_other] = False
                
            front_y = y_bootstrapped[nondom_mask]
            
            if len(front_y) == 0:
                continue

            ref_point = context["ref_point"]
                
            def hypervolume(points, reference):
                # Simple implementation assuming all objectives are maximized
                volumes = []
                for p in points: 
                    vol = np.prod(np.maximum(reference - p, 0))
                    if not any(vol == v and (p >= q).all() and (q > p).any()
                               for q,v in zip(points[:points.index(p)] + points[1+points.index(p):], volumes)):
                        # Avoid double counting same volume due to numerical precision
                        volumes.append(vol)
                return sum(volumes) if len(set([v for v, _ in enumerate(volumes)]) ) > 0 else float('nan')
                
            vol_before = hypervolume(front_y, ref_point)

            front_with_cand = np.vstack((front_y, y_cand_pred))
            
            # Compute the new volume after adding candidate
            if len(y_bootstrapped) == 1 and (y_bootstrapped[0] != y_cand_pred).any():
                vol_after_no_fronts_added = hypervolume([np.array(y_cand_pred)], ref_point)
                
                improvement_per_candidate_addition = (
                    vol_after_no_fronts_added - 
                     np.prod(np.maximum(ref_point, 1e-9) + (y_cand_pred > reference).astype(float))
                )
            else:
                 # For multiple points
                try:  
                    vol_after_with_cand = hypervolume(front_with_cand, ref_point)
                    
                    if not isinstance(vol_before, float):
                        continue
                        
                    improvement_per_candidate_addition = max(0.0,
                                                             (vol_after_with_cand - 
                                                              np.sum(np.maximum(ref_point[None]  , front_y).prod(axis=1))))
                except Exception:
                     # fallback
                    vol_after_no_fronts_added = hypervolume([np.array(y_cand_pred)], ref_point)
                    
                    improvement_per_candidate_addition = max(0.0,
                                                             (vol_after_no_fronts_added - 
                                                              np.prod(np.maximum(ref_point, 1e-9) +  
                                                                      (y_cand_pred > reference).astype(float))))

            total_improvement += float(improvement_per_candidate_addition)
            
        avg_improvement = total_improvement / n_draws if n_valid_draws !=0 else -float('inf')
        
        scores.append(avg_improvement)

    return [max(1e-9, s) for s in scores]