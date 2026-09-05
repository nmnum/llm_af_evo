def score_pool(context):
    """Score candidates by expected hypervolume improvement adjusted for uncertainty and progress-aware exploration."""
    names = context["objective_names"]
    ref_point = np.array(context["ref_point"])
    
    # Use the pre-computed acquisition value as baseline quality estimate  
    acq_values = [cand['acq_value_norm'] for cand in context["pool"]]
    
    # Normalize acquisition values to [0, 1] across pool
    a_min, a_max = min(acq_values), max(acq_values)
    if abs(a_max - a_min) < 1e-9:
        norm_acqs = np.array([0.5] * len(acq_values))
    else:  
        norm_acqs = (np.array(acq_values) - a_min) / (a_max - a_min)

    # Progress-aware uncertainty adjustment
    campaign_progress = context["campaign"]["progress"]
    
    scores = []
    for i, cand in enumerate(context["pool"]):
        
        gp = cand['gp_posterior']
            
        # Compute total normalized std across objectives  
        sigma_norm_sum = sum(gp[name]["std"]/context["pareto_front_range"][name] 
                             for name in names)
                             

        # Progress-aware weighting: early exploration, later exploitation
        w_explore = 0.5 + (1 - campaign_progress) * 0.3   # Decrease explore weight over time
        
        score_i = norm_acqs[i]
        
        if sigma_norm_sum > 0:
            ucb_component = w_explore * sigma_norm_sum 
            score_i += max(0., min(ucb_component, 1.)) 
            
        scores.append(score_i)
    
    return scores