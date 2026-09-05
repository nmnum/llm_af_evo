def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    
    # Check if obj_correlation is provided and not empty
    if context.get("obj_correlation", {}): 
        names = context["objective_names"]
        
        # Build list of all pairwise negative correlations for each candidate  
        corr_bonus_list = []  # one entry per cand, sum of neg corrs
        
        for i in range(len(context['pool'])):
            total_neg_corr = 0.0
            count = 0
            
            for name_a in names: 
                for name_b in names:
                    if name_a != name_b:
                        key = f"{name_a},{name_b}"
                        
                        # Only process keys that actually exist (not all pairs need to be present)
                        corr_val_list = context["obj_correlation"].get(key, [])
                        if i < len(corr_val_list):  # ensure index is valid
                            val = corr_val_list[i]
                            
                            # Add negative correlations positively 
                            if val < -0.1:   # threshold for "negative enough"
                                total_neg_corr += abs(val)
                                count += 1
            
            avg_neg_corr = total_neg_corr / max(count, 1)    # avoid div by zero
            corr_bonus_list.append(avg_neg_corr)

        bonus_weight = 0.05     # small weight on the correlation signal

        for i, cand in enumerate(context["pool"]):
            
            base_acq_val = cand['acq_value_norm']
            bonus_term = bonus_weight * corr_bonus_list[i]
          
            scores.append(base_acq_val + bonus_term)
    else:
        
         # Return baseline acquisition values when no correlation info available
        for cand in context["pool"]:
             scores.append(cand['acq_value_norm'])
    
    return scores