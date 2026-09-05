def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context.get('obj_correlation', {}):
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    neg_corr_bonus = []

    # Precompute negative correlation bonuses
    for i, _ in enumerate(context["pool"]):
        bonus = 0.0
        count = 0
        for key in corr_keys:
            if ',' not in key: continue
            a_name, b_name = key.split(',')
            if a_name not in names or b_name not in names: continue
            
            val = context['obj_correlation'][key][i]
            # Only consider negative correlations as beneficial bonuses
            if val < 0:
                bonus += -val 
                count += 1
                
        neg_corr_bonus.append(bonus / max(count, 1))

    for i, cand in enumerate(context["pool"]):
        base_score = cand['acq_value_norm']
        corr_bonus = neg_corr_bonus[i]
        
        # Blend with a small weight
        score = base_score + 0.2 * corr_bonus 
        scores.append(score)
    
    return scores