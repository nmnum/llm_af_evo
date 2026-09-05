def score_pool(context):
    """Blend acquisition value with uncertainty-aware progress tracking to favor candidates that are both promising and sufficiently novel."""
    
    if not context["pool"]:
        return []
        
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    front_range = context["pareto_front_range"] 
    campaign = context["campaign"]

    # Compute a dynamic weight that balances exploitation and exploration
    progress = campaign['progress']
    
    if progress < 0.3:
        w_exploit = 0.2   # Early stage: more focus on exploring new regions  
    elif progress < 0.7:
        w_exploit = 0.5 
    else:
        w_exploit = 0.8   # Later stages: exploit promising areas

    scores = []
    
    for cand in context["pool"]:
        
        gp_posterior = cand["gp_posterior"]
                
        acq_norm = cand['acq_value_norm']
            
        # Estimate uncertainty as normalized sum of stds
        sigma_sum = 0.0 
        for name in names:
            sigma_sum += gp_posterior[name]["std"] / front_range[name]
        
        # Normalize the total uncertainty to [0,1] based on observed range  
        ucb_weight = w_exploit + (1 - w_exploit) * np.tanh(sigma_sum)
            
        score = acq_norm  + ucb_weight * sigma_sum
        
        scores.append(score)

    return scores