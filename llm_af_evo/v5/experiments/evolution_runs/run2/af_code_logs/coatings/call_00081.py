def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context.get("obj_correlation", {}):
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    
    for i, cand in enumerate(context["pool"]):
        acq_val = cand["acq_value_norm"]
        
        bonus = 0.0
        if corr_keys:  
            neg_corr_count = 0
            total_neg_corr = 0.0
            
            for key in corr_keys:
                val = context["obj_correlation"][key][i]
                if val < -0.1: 
                    total_neg_corr += val
                    neg_corr_count += 1
                    
            bonus = (total_neg_corr / max(neg_corr_count, 1)) * 0.05
            
        scores.append(acq_val + bonus)
        
    return scores