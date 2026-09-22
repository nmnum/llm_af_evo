def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context.get("obj_correlation", {}):
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    
    # Build a map from name_a,name_b to the correlation value
    corrs_by_pair = {}
    for key in corr_keys:
        if "," not in key or len(key.split(",")) != 2:
            continue
        n1, n2 = key.split(",")
        if n1 in names and n2 in names:
            corrs_by_pair[key] = context["obj_correlation"][key]
    
    for i, cand in enumerate(context["pool"]):
        acq_val = cand["acq_value_norm"]
        
        # Compute average negative correlation bonus
        corr_bonus = 0.0
        total_weight = 0
        
        gp_posterior = cand["gp_posterior"]
        means = [gp_posterior[name]["mean"] for name in names]
        
        if not corrs_by_pair:
            scores.append(acq_val)
            continue
            
        # For each pair of objectives, compute the correlation bonus
        n_obj = len(names)
        total_corr_bonus = 0.0
        
        for key, corr_values in corrs_by_pair.items():
            try:
                idx1, idx2 = names.index(key.split(",")[0]), names.index(key.split(",")[1])
                
                # If this candidate's mean on one objective is high (near front), 
                # and the other low (not well covered by current pareto),
                # then a positive correlation in that direction gives bonus
                if means[idx2] > 0.5:   # assuming normalized, so above median suggests good coverage  
                    corr_bonus += -corr_values[i]
                    total_weight += abs(corr_values[i])
            except (ValueError):
                continue
                
        avg_corr = corr_bonus / max(total_weight, 1e-8)
        
        final_score = acq_val + 0.2 * avg_corr
        scores.append(final_score)

    return scores