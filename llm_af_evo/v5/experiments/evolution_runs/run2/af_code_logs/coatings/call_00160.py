def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context.get("obj_correlation", {}):
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    neg_corr_bonus = 0.0
    
    # Precompute the correlation bonus per candidate
    for i, cand in enumerate(context["pool"]):
        total_neg_corr = sum(
            val 
            for key in corr_keys 
            if (val := context["obj_correlation"][key][i]) < 0  
        )
        
        neg_corr_bonus = -total_neg_corr / max(len(corr_keys),1)
        scores.append( cand["acq_value_norm"] + 0.2 * neg_corr_bonus )

    return scores