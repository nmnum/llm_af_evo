def score_pool(context):
    """Blend hypervolume acquisition value with a correlation-based bonus term when objective correlations are available."""
    scores = []
    acq_values = [cand["acq_value_norm"] for cand in context["pool"]]
    
    if not context.get("obj_correlation", {}):  # No correlation signal
        return acq_values
    
    names = context["objective_names"]
    bonus_terms = np.zeros(len(context["pool"]))
    
    corr_key_map = {}
    for key, values in context["obj_correlation"].items():
        a, b = key.split(",")
        if (a, b) not in corr_key_map:
            corr_key_map[(a, b)] = []
        corr_key_map[(a, b)].extend(values)
    
    # For each candidate
    for i_cand in range(len(context["pool"])):
        bonus_sum = 0.0
        n_pairs = 0
        
        if len(names) < 2:
            continue
            
        # Compute average negative correlation involving this candidate's objectives  
        for a, b in corr_key_map.keys():
            try: 
                idx_a = names.index(a)
                idx_b = names.index(b)
                
                val_corr = corr_key_map[(a,b)][i_cand]
            
                if (val_corr < 0): # only consider negative correlations
                    bonus_sum += -1. * val_corr  
                    
                n_pairs += 1
                
            except ValueError: 
                continue  # Skip invalid objective names

        avg_neg_correlation = bonus_sum / max(n_pairs, 1) 
        
        if not np.isnan(avg_neg_correlation):
            bonus_terms[i_cand] = avg_neg_correlation
    
    final_scores = [a + b * 0.2 for a,b in zip(acq_values,bonus_terms)]
    
    return final_scores