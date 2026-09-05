def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context.get('obj_correlation', {}):
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    
    # Precompute correlation bonus per candidate
    bonuses = np.zeros(len(context["pool"]))
    if corr_keys:
        num_corr_terms = 0
        for key in corr_keys:
            a, b = key.split(',')
            if a not in names or b not in names:
                continue
            
            corrs = context['obj_correlation'][key]
            # Only consider negative correlations as bonus contributors (as per prompt)
            bonuses += np.array(corrs) * (np.array(corrs) < 0)
            num_corr_terms += 1
        
        if num_corr_terms > 0:
            bonuses /= num_corr_terms
    
    return [cand['acq_value_norm'] + 0.25 * bonus for cand, bonus in zip(context["pool"], bonuses)]