def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context["obj_correlation"]:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    # Extract correlation values per candidate
    correlations = np.array([context["obj_correlation"][key][i] 
                            for i in range(len(context["pool"])) 
                            for key in corr_keys])
    
    # Reshape to (n_candidates, n_pairs)
    num_pairs = len(corr_keys)  
    corrs = correlations.reshape(-1, num_pairs)

    bonus_terms = []
    pair_idx = 0
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        
        # For each candidate compute the average negative correlation with objectives that are already well-covered 
        neg_corr_sum = 0.0  
        count_neg_corrs = 0
        
        for j, name_a in enumerate(names):            
            for k, name_b in enumerate(names):
                if j >= k: continue
                key = f"{name_a},{name_b}"
                
                corr_val = corrs[i][pair_idx]
                    
                # Bonus is added only when correlation value is negative 
                neg_corr_sum += max(0.0, -corr_val)
                count_neg_corrs += 1 if corr_val < 0 else 0
                
                pair_idx += 1

        bonus_term = (neg_corr_sum / count_neg_corrs) if count_neg_corrs > 0 else 0
        bonus_terms.append(bonus_term)

    # Combine acquisition value and the small correlation-based bonus 
    alpha = 0.2  
    for i, cand in enumerate(context["pool"]):
        score = cand["acq_value_norm"]
        
        adjusted_score = score + (alpha * bonus_terms[i])
        scores.append(adjusted_score)
    
    return scores