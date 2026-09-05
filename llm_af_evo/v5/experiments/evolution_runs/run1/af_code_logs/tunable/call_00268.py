def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context["obj_correlation"]:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    
    # Precompute correlation bonuses
    bonus_terms = np.zeros(len(context["pool"]))
    if corr_keys:
        for key in corr_keys:
            a, b = key.split(",")
            vals_a_b = context["obj_correlation"][key]
            # Bonus is the average of negative correlations (i.e., encourage progress on objectives that are negatively correlated with current front)
            bonus_terms += np.array(vals_a_b) * (-1.0 if "neg" in str(np.mean(vals_a_b)) else 1.0)

    for i, cand in enumerate(context["pool"]):
        base_score = cand["acq_value_norm"]
        # Add a small correlation-based bonus (e.g., multiply by 0.5 to make it secondary)
        adjusted_score = base_score + 0.2 * bonus_terms[i]
        scores.append(adjusted_score)

    return scores