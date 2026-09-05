def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context["obj_correlation"]:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    
    for i, cand in enumerate(context["pool"]):
        acq_val = cand["acq_value_norm"]
        
        bonus = 0.0
        if corr_keys:  
            num_corr_terms = 0
            for key in corr_keys:
                a_name, b_name = key.split(",")
                if (a_name in names and b_name in names):
                    corr_vals = context["obj_correlation"][key]
                    if i < len(corr_vals): 
                        bonus += max(0.0, -corr_vals[i])  
                        num_corr_terms += 1
            if num_corr_terms > 0:
                bonus /= num_corr_terms
        
        scores.append(acq_val + 0.1 * bonus)
    
    return scores