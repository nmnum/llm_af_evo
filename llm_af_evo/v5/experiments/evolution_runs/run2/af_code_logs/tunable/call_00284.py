def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context["obj_correlation"]:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    
    # Precompute all correlation values
    correlations = np.array([context["obj_correlation"][key] for key in corr_keys]).T  # (n_pool, n_corr_pairs)
    
    for i, cand in enumerate(context["pool"]):
        acq_val = cand["acq_value_norm"]
        
        if len(corr_keys) == 0:
            bonus = 0.0
        else:
            neg_corrs = [cand_vals[j] for j, key in enumerate(corr_keys)
                         if (key.split(',')[0], key.split(',')[1]) != ('obj_1', 'obj_2') or 
                            cand_vals[0]*cand_vals[1]< 0]
            
            bonus = np.mean(neg_corrs) * 0.05
        
        scores.append(acq_val + bonus)
    
    return scores