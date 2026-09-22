def score_pool(context):
    """Blend acquisition value with a correlation-based bonus to reward progress on negatively correlated objectives."""
    if not context["obj_correlation"]:
        return [cand['acq_value_norm'] for cand in context['pool']]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    
    # Precompute correlation bonuses
    bonus_terms = np.zeros(len(context["pool"]))
    if len(corr_keys) > 0:
        for key in corr_keys:
            a, b = key.split(',')
            if a not in names or b not in names: continue  
            vals = context['obj_correlation'][key]
            # Use negative correlations as bonus signal
            bonus_terms += np.array(vals)
    
    avg_bonus = bonus_terms / len(corr_keys) if corr_keys else 0.0
    
    for i, cand in enumerate(context["pool"]):
        base_score = cand['acq_value_norm']
        bonus = max(0., -avg_bonus[i]) * 0.1
        scores.append(base_score + bonus)
    
    return scores