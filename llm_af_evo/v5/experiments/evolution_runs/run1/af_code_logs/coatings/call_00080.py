def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context.get("obj_correlation", {}):
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    
    # Precompute correlation bonuses
    bonus_values = np.zeros(len(context["pool"]))
    if corr_keys:
        for key in corr_keys:
            vals = np.array(context["obj_correlation"][key])
            a, b = key.split(",")
            idx_a = names.index(a)
            idx_b = names.index(b)
            
            # Bonus is positive when correlation is negative and objectives are on front
            bonus_values += (vals < 0) * vals
    
    for i, cand in enumerate(context["pool"]):
        base_score = cand["acq_value_norm"]
        corr_bonus = 0.1 * bonus_values[i] 
        scores.append(base_score + corr_bonus)
    
    return scores