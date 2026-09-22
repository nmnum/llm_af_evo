def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    acq_values = [cand["acq_value_norm"] for cand in context["pool"]]
    
    if not context.get("obj_correlation", {}):
        return acq_values
    
    names = context["objective_names"]
    corr_bonus = np.zeros(len(context["pool"]))
    
    # Collect correlation values between objectives
    for key, vals in context["obj_correlation"].items():
        a_name, b_name = key.split(",")
        if a_name not in names or b_name not in names:
            continue
        
        idx_a = names.index(a_name)
        idx_b = names.index(b_name)
        
        # For each candidate: compute average negative correlation
        for i, corr_val in enumerate(vals):
            if corr_val < 0 and (i < len(corr_bonus)):
                corr_bonus[i] += -corr_val
    
    # Normalize bonus to match scale of acq_values (~[0,1])
    max_corr = np.max(np.abs(corr_bonus)) + 1e-8
    normalized_bonus = corr_bonus / max_corr * 0.1

    return [acq + b for acq, b in zip(acq_values, normalized_bonus)]