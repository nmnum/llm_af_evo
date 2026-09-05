def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    
    # Check if obj_correlation is provided and not empty
    if context.get('obj_correlation', {}): 
        names = context["objective_names"]
        
        # Prepare the list of candidate-wise correlation values for each pair
        corr_bonus = [0.0] * len(context['pool'])
        total_pairs = 0
        
        # Iterate through all available correlations in obj_correlation (only those that exist)
        for key, vals in context["obj_correlation"].items():
            if "," not in key:
                continue
            
            name_a, name_b = key.split(",", 1) 
            if name_a not in names or name_b not in names:  
                # Skip invalid objective pair keys
                continue
                
            total_pairs += 1 
            
            for i, val in enumerate(vals):
                corr_bonus[i] += max(0.0, -val)
        
        # Average the bonus across pairs and scale it down (e.g., by a factor of 2) 
        if total_pairs > 0:
           norm_corr = [b / float(total_pairs)/2 for b in corr_bonus]
            
    else:  
       # If no correlation signal, use zero bonuses
       norm_corr = [0.0] * len(context['pool'])
    
        
    acq_values = [cand["acq_value_norm"] for cand in context["pool"]]
    
    
    scores = [
        a + b 
         for (a,b) in zip(acq_values,norm_corr)
     ]
     
    return scores