def modifier(context):
    """Reward candidates that are diverse in objective space relative to selected points, encouraging exploration away from crowded regions."""
    pool = context["pool"]
    values = [0.0] * len(pool)
    
    # Use a fixed reference point for diversity calculation (e.g., worst observed per obj)  
    ref_point = np.array([context['ref_point_by_name'][name] for name in context['objective_names']])
    
    if 'X_obs' not in context or len(context["X_obs"]) == 0:
        # No observations yet, return neutral correction
        return values

    selected_y = []
    observed_Y = context["Y_obs"]
    objective_names = context["objective_names"]

    for i in range(len(pool)):
        cand_x = pool[i]["x"] 
        gp_posterior = pool[i]['gp_posterior']
        
        # Estimate candidate's mean objectives
        means = np.array([gp_posterior[name]["mean"] for name in objective_names])
            
        min_hv_improvement = float('inf')
        hv_ref_point = ref_point.copy()
    
        if len(selected_y) > 0:
            selected_means = []
            # Get the current Pareto front's mean objectives
            pareto_front_Y = context["pareto_front"]
                
            for y in observed_Y: 
                dominates_any_selected_pareto = False  
                for pf_point in pareto_front_Y:
                    if np.all(pf_point >= y) and not np.array_equal(pf_point, y):
                        dominates_any_selected_pareto = True
                        break
                        
                # Only consider non-dominated points as potential contributors to HV expansion 
                if not dominates_any_selected_pareto:  
                    selected_means.append(y)
                
            for j in range(len(selected_y)):
                 sel_mean_obj = np.array([selected_y[j][name] for name in objective_names])
                 
                 hypervolume_improvement_estimate = 1.0
                 # Compute the change to HV if this candidate were added 
                 temp_ref_point = hv_ref_point.copy()
                  
                 for k, obj_name in enumerate(objective_names):
                     improved_obj_value = max(means[k], sel_mean_obj[k])
                     
                     hypervolume_improvement_estimate *= (temp_ref_point[k] - improved_obj_value)
                     

             if min_hv_improvement > hv_imp:
                # Use the smallest HV improvement as a proxy for how much this candidate adds
                 min_hv_improvement =hv_imp  

        values[i] += 0.1 * max(0, (min_hv_improvement - np.mean(selected_means) if len(selected_means)>0 else 0)) 

    return values