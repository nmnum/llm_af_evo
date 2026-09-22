def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    acq_values = [cand["acq_value_norm"] for cand in context["pool"]]
    
    if not context.get("obj_correlation", {}):  # No correlation signal
        return acq_values
    
    names = context["objective_names"]
    bonus_terms = np.zeros(len(context["pool"]))
    
    corr_keys = [k for k in context["obj_correlation"].keys() if "," in k]
    for key in corr_keys:
        obj_a, obj_b = key.split(",")
        if obj_a not in names or obj_b not in names:
            continue
        correlations = np.array(context["obj_correlation"][key])
        
        # Bonus: negative correlation contributes more (inverse of positive)
        bonus_terms += -np.minimum(correlations, 0)  
    
    # Normalize and scale the bonus term to be small relative to acquisition value 
    if not(np.all(bonus_terms == 0)):
        max_bonus = np.max(bonus_terms)
        min_bonus = np.min(bonus_terms)
        normalized_bonuses = (bonus_terms - min_bonus)/(max_bonus-min_bonus) if max_bonus != min_bonus else bonus_terms
        scaled bonuses=normalized_bonuses * 0.1 # small weight compared to acquisition value 
    else:
       scaled_bonuses=np.zeros(len(context["pool"]))
    
    scores=[acq + bonus for acq,bonus in zip(acq_values,scaled_bonuses)]
   
    return scores