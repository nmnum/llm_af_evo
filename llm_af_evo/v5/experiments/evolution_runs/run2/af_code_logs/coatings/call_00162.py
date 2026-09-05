def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context["obj_correlation"]:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    # Extract correlation values per candidate
    corrs_by_cand = np.array([context["obj_correlation"][key] 
                              for key in corr_keys]).T  # shape: (n_candidates, n_corr_pairs)
    
    for i, cand in enumerate(context["pool"]):
        acq_val = cand["acq_value_norm"]
        
        bonus = 0.0
        if corrs_by_cand[i].size > 0:
            corr_vals = corrs_by_cand[i]
            # Weight negative correlations positively (bonus for improving negatively correlated objectives)
            neg_corr_bonus = np.mean(np.clip(corr_vals, None, 0)) * -1 
            bonus = max(0.0, neg_corr_bonus) / len(names)

        scores.append(acq_val + 0.25 * bonus)
    
    return scores