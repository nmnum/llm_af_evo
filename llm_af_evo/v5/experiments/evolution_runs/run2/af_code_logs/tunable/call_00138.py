def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    acq_values = [cand["acq_value_norm"] for cand in context["pool"]]
    
    if not context.get("obj_correlation", {}):
        return acq_values
    
    names = context["objective_names"]
    corr_bonus = np.zeros(len(context["pool"]))
    
    # Collect correlation values between pairs involving objectives
    for key, vals in context["obj_correlation"].items():
        a, b = key.split(",")
        if a not in names or b not in names:
            continue  # Skip irrelevant correlations
        
        idx_a = names.index(a)
        idx_b = names.index(b)
        
        corr_vals = np.array(vals) 
        neg_corr_mask = corr_vals < 0
        bonus_contribution = -np.mean(corr_vals[neg_corr_mask]) if np.any(neg_corr_mask) else 0.0
        
        # Accumulate the average negative correlation contribution for each candidate  
        corr_bonus += bonus_contribution / len(names)
    
    return acq_values + 0.1 * corr_bonus