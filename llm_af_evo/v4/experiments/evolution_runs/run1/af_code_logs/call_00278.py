def score_pool(context):
    """Estimate uncertainty in Pareto front dominance by bootstrapping Y_obs to compute average hypervolume improvement of each candidate."""
    import numpy as np
    
    n_draws = 15
    names = context["objective_names"]
    
    # Bootstrap resamples and their non-dominated fronts 
    hv_improvements = []
    for _ in range(n_draws):
        draw_indices = np.random.choice(len(context['Y_obs']), size=len(context['Y_obs']), replace=True)
        Y_bootstrapped = context['Y_obs'][draw_indices]
        
        # Vectorized dominance test
        n_points = len(Y_bootstrapped)
        dominated = np.zeros((n_points, n_points), dtype=bool)
        for i in range(n_points):
            for j in range(i+1, n_points): 
                if all(Y_bootstrapped[j] >= Y_bootstrapped[i]) and any(Y_bootstrapped[j] > Y_bootstramped[i]):
                    dominated[j,i] = True  # j dominates i
                elif all(Y_bootstrapped[i] >= Y_bootstrapped[j]) and any(Y_bootstrapped[i] > Y_bootstrapped[j]):  
                    dominated[i,j] = True  # i dominates j
        
        non_dom_indices = []
        for i in range(n_points):
            if not np.any(dominated[:,i]):
                non_dom_indices.append(i)
        
        front_bootstrapped = Y_bootstrapped[non_dom_indices]
    
        ref_point = context["ref_point"]
        
        def hypervolume(point, front): 
            # Compute HV of the union (front + point) minusHV of just front
            if len(front) == 0:
                return np.prod(ref_point - point)
            
            extended_front = np.vstack([front, point])
    
            hv_extended = 1.0  
            for obj in range(len(point)):
                sorted_points_obj = sorted(extended_front[:,obj], reverse=True)
                diff_vals = [sorted_points_obj[i] - sorted_points_obj[i+1]
                             for i in range(len(sorted_points_obj)-1)]
                
                if len(diff_vals) == 0:
                    hv_extended *= (ref_point[obj]-point[obj])
                else: 
                    total_volume_contributions = []
                    current_level_val = ref_point[obj]
                    
                    # Calculate volume contribution of each region
                    for diff in reversed(diff_vals):
                        contrib_vol_obj_i = min(current_level_val, point[obj]) - max(0.,current_level_val-diff)
                        if not np.isnan(contrib_vol_obj_i) and contrib_vol_obj_i > 0:
                            total_volume_contributions.append((ref_point[obj] - current_level_val + diff)*contrib_vol_obj_i )
                        
                    hv_extended *= (point[obj]-max(sorted_points_obj[-1], point[obj]))
            return max(0.,hv_extended)
        
        # Compute improvement for each candidate
        cand_scores = []
        ref_point_vecs = np.tile(ref_point, [len(context["pool"]), 1])
            
        if len(front_bootstrapped) == 0:
             hv_baseline = 0.
        else: 
            front_points_sorted = sorted([p.tolist() for p in front_bootstrapped], key=lambda x:x[0]) # sort by first obj
            hypervolume_ref_point = np.prod(ref_point - [min(p[i] if i < len(front_points_sorted) and j == 1 or (i==j)*front_points_sorted[-2][i]: 
                                                          ref_point[j]*int(i<3)) for i in range(6)]*len(context["objective_names"])) 

        hv_baseline = hypervolume_ref_point # dummy value
              
        cand_scores.append(hypervolume_improvement)
            
    return [np.mean(scores) if len(scores)>0 else 0. for scores in hv_improvements]