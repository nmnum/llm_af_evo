def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context["obj_correlation"]:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    
    for i, cand in enumerate(context["pool"]):
        acq_val = cand["acq_value_norm"]
        
        if not corr_keys:
            bonus = 0.0
        else:
            total_corr = sum(
                context["obj_correlation"][key][i] 
                for key in corr_keys 
                if all(k in names for k in key.split(","))
            )
            
            avg_corr = total_corr / len(corr_keys) if corr_keys else 0.0
            bonus = max(0, -avg_corr * 0.1)
        
        scores.append(acq_val + bonus)

    return scores