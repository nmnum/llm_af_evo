def score_pool(context):
    """Blend acquisition value with an uncertainty-weighted front proximity signal that encourages exploring under-covered regions of objective space based on how much improvement a candidate could offer if its predictions were certain."""
    names = context["objective_names"]
    ref_point = context["ref_point"]
    pareto_front = context["pareto_front"]
    
    # Compute the hypervolume contribution for each point in Pareto front
    hv_contributions = []
    if len(pareto_front) > 0:
        for i, pf_pt in enumerate(pareto_front):
            vol_diff = np.prod(np.maximum(pf_pt - ref_point, 0))
            hv_contributions.append(vol_diff)
    
    # For each candidate compute a weighted front proximity score
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        pred_means = [gp_posterior[name]["mean"] for name in names]
        stds = np.array([gp_posterior[name]["std"] for name in names])
        mean_vec = np.array(pred_means)
    
        # Compute normalized uncertainty (1.0 is maximum, 0.0 no noise at all)
        front_range_vals = [context["pareto_front_range"][name] for name in names]
        if any(r == 0 for r in front_range_vals):
            sigma_norm = np.sum(stds / max(front_range_vals)) # Avoid division by zero
        else:
            sigma_norm = np.mean(np.divide(stds, front_range_vals))
        
        hv_improvement_if_certain = None
        
        if len(pareto_front) > 0 and not all(s == 1. for s in stds):
            
            min_dist_to_pf = float('inf')
    
            # Compute distance to nearest Pareto point (using means)
            mean_vec_array = np.array(pred_means).reshape(1, -1)

            distances_sq = ((mean_vec_array[:, None] - pareto_front[None]) ** 2.).sum(axis=2) 
            
            min_dist_to_pf = np.sqrt(distances_sq.min())

        else:
           # No front or no uncertainty: assume max improvement
           hv_improvement_if_certain = float('inf')

        
       if hv_improvement_if_certain is None:

            weight_factor = 1.0 / (sigma_norm + 1e-8)   # higher sigma, lower factor

            
            score_base = cand["acq_value_norm"] 

           
            scores.append(score_base * weight_factor)
        else:
           score_base = cand["acq_value_norm"]
          
           if np.isinf(hv_improvement_if_certain):
               weights_to_use = 1.0
           elif hv_improvement_if_certain > 5e-4: 
                # Adjust based on the relative size of HV contribution (larger means more potential)
                
                rel_hypervolume_gain = np.log(hv_improvement_if_certain + 2.) / (
                    max(np.sum(np.abs(ref_point - pareto_front.min(axis=0))), .1) if len(pareto_front) > 0 else 1.)
    
                weights_to_use = (rel_hypervolume_gain ** (-.5)) * (.8)
           else:
               # Low improvement: penalize uncertainty
               weight_factor_penalty_for_uncertainty = np.exp(-sigma_norm / .2)

               scores.append(score_base * max(0., weight_factor_penalty_for_uncertainty))

    return [max(s, 1e-6) for s in scores]