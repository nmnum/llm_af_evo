def score_pool(context):
    """Blend acquisition value with progress-aware uncertainty and rescaled hypervolume bonus."""
    if not context["pool"]:
        return []
    
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    front_range = context["pareto_front_range"]
    campaign = context["campaign"]

    # Normalize uncertainty by the observed range
    def normalized_uncertainty(cand):
        return sum(cand['gp_posterior'][name]['std'] / front_range[name]
                   for name in names) / len(names)

    scores = []
    
    base_acq_values = [cand['acq_value_norm'] for cand in context["pool"]]
    
    # Use a decaying factor based on campaign progress to shift balance between exploitation and exploration
    w_exploit = 0.7 + (1 - campaign.get("progress", 0)) * 0.3
    
    max_acq_val = np.max(base_acq_values)
    min_acq_val = np.min(base_acq_values)

    # Normalize acq values to [0, 1] range for consistent blending
    if abs(max_acq_val - min_acq_val) < 1e-8:
        normed_base_scores = [0.5]*len(context["pool"])
    else: 
        normed_base_scores = [(v-min_acq_val)/(max_acq_val-min_acq_val)
                              for v in base_acq_values]

    # Compute hypervolume bonus from the current pool
    hv_bonus_terms = []
    
    if len(context['pareto_front']) > 0:
        
        for cand in context["pool"]:
            gp_posterior = cand["gp_posterior"]
            
            predicted_obj_vals = [gp_posterior[name]["mean"] 
                                  for name in names]
              
            # Compute the hypervolume contribution by extending front with this candidate
            extended_front = np.vstack((context['pareto_front'], predicted_obj_vals))
          
            try:
                hv_extended = hypervolume(extended_front, ref_point)
                hv_current  = hypervolume(context["pareto_front"], ref_point)  
                
                if not (hv_extended <= hv_current):
                    # Use difference as bonus term
                    hv_bonus_terms.append(hv_extended - hv_current)
                    
            except:
                 pass
                
        max_hv_bonuses = np.max([0.] + hv_bonus_terms)

    else: 
         max_hv_bonuses = 1.0

    for i, cand in enumerate(context["pool"]):
        
        base_acq_val   = normed_base_scores[i]
        unc            = normalized_uncertainty(cand)
  
        # Combine exploitation and uncertainty
        score_exploit_plus_uncert = w_exploit * (base_acq_val) + \
                                    ((1 - w_exploit)*unc)

       if len(hv_bonus_terms)>0:
           hv_bonuses  = [b/max_hv_bonuses for b in hv_bonus_terms]
           
            # Apply bonus as a scaling factor
           score_with_bonus=score_exploit_plus_uncert * (1. + np.clip(2.*hv_bonuses[i], 
                                                                      0., .5))
       else:
             score_with_bonus = score_exploit_plus_uncert

        scores.append(score_with_bonus)

    return [max(s, 1e-6) for s in scores]

def hypervolume(front, ref_point):
    """Compute the two-dimensional hypervolume of a front relative to reference point."""
    
    if len(front.shape) == 1:
       front = np.expand_dims(front,axis=0)
        
    n_points = len(front)

    vol_total = float(0.0)

    for i in range(n_points):
        x_i, y_i = front[i][0], front[i][1]
    
         # Simple calculation of hypervolume contribution relative to ref_point
        if (x_i < ref_point[0]) and (y_i < ref_point[1]):
            vol_total += float((ref_point[0] - x_i) * (ref_point[1]- y_i))

    return max(vol_total, 0.0)