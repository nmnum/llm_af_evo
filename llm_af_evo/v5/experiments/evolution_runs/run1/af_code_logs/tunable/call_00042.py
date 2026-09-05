def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context["obj_correlation"]:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    
    # Precompute all correlation values per candidate
    bonus_terms = np.zeros(len(context["pool"]))
    if corr_keys:
        for key in corr_keys:
            a, b = key.split(',')
            if a not in names or b not in names: continue  # Skip invalid keys
            vals_a_b = context['obj_correlation'][key]
            # Only consider negative correlations as beneficial bonuses (positive ones are ignored)
            bonus_terms += np.array(vals_a_b) * (-1.0)

    # Normalize the bonus terms to be small relative to acquisition value, scale by 0.1
    max_bonus = np.max(bonus_terms) if len(bonus_terms[bonus_terms > 0]) else 1e-6
    normalized_bonus = (bonus_terms / max_bonus * 0.1)

    # Combine bonus with base acq_value_norm to form final score
    for i, cand in enumerate(context["pool"]):
        scores.append(cand['acq_value_norm'] + normalized_bonus[i])
    
    return scores