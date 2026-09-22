def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    
    # Check if obj_correlation is provided and not empty
    if context.get("obj_correlation", {}):
        names = context["objective_names"]
        
        # Prepare the list of candidate indices for each pair in correlation dict  
        corr_keys = [k.split(",") for k in context['obj_correlation'].keys()]
        
        # Identify which pairs involve objectives from current front
        pf_range = {name: context["pareto_front_range"][name] for name in names}
    
        bonus_terms = []
        for i, cand in enumerate(context["pool"]):
            gp_posterior = cand["gp_posterior"]
            
            corr_vals_list = [context['obj_correlation'][k][i]
                              for k in context['obj_correlation'].keys()
                              if any(name not in names or name == "" 
                                     for name in k.split(",")) is False]  # ensure valid keys
            
            bonus_val = sum(corr_vals_list) / len(corr_vals_list) if corr_vals_list else 0.0
            bonus_terms.append(bonus_val)
        
        acq_norms = [cand["acq_value_norm"] for cand in context["pool"]]
            
    else:
       # Return baseline score (just acquisition value normed to [0,1])
       return [cand['acq_value_norm'] for cand in context['pool']]
    
    final_scores = []
    base_weight = 0.95
    bonus_weight = 0.05
    
    for i in range(len(context["pool"])):
        score_i = (base_weight * acq_norms[i] + 
                   bonus_weight * max(0, -bonus_terms[i])) # Bonus only helps if negative correlation exists
        
        final_scores.append(score_i)
    
    return final_scores