def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    
    # Check if obj_correlation is populated (non-empty dict)
    has_corr_signal = bool(context.get('obj_correlation', {}))
    
    for cand in context["pool"]:
        acq_val = cand['acq_value_norm']
        
        corr_bonus = 0.0
        if has_corr_signal:
            # Build correlation bonus from negative correlations with current front objectives 
            names = context["objective_names"]
            
            # Collect all relevant per-candidate correlation values for this candidate  
            neg_corrs_sum = 0.
            num_neg_corrs = 0
            
            corr_dict = context['obj_correlation']
        
            for i, name_a in enumerate(names):
                for j, name_b in enumerate(names): 
                    if i >= j: continue
                    key = f"{name_a},{name_b}"
                    
                    # Check that the correlation signal exists and corresponds to this candidate  
                    corr_vals = corr_dict.get(key)
                    if not (corr_vals is None or len(corr_vals) <= 0):
                        try:
                            val = float(corr_vals[cand['idx']])
                            neg_corrs_sum += max(0., -val)
                            num_neg_corrs += 1
                        except Exception: 
                            pass # skip malformed entries
            
            if num_neg_corrs > 0:
                avg_neg_corr = neg_corrs_sum / num_neg_corrs  
                corr_bonus = min(avg_neg_corr, .2) * acq_val
        
        scores.append(acq_val + corr_bonus)
        
    return scores