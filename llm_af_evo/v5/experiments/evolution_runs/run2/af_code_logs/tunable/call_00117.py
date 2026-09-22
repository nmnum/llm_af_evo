def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    
    # Check if obj_correlation is provided and not empty
    has_corr_signal = bool(context.get('obj_correlation', {}))
    
    for cand in context["pool"]:
        acq_value = cand["acq_value_norm"]
        
        corr_bonus = 0.0
        
        if has_corr_signal:
            # Collect correlation values that are negative (indicating beneficial trade-offs)
            neg_corr_values = []
            
            names = context['objective_names']
            for i, name_a in enumerate(names):
                for j, name_b in enumerate(names):
                    if i >= j:  # Avoid duplicate pairs and self-correlations
                        continue
                    
                    key = f"{name_a},{name_b}"
                    
                    corr_val = None 
                    try:
                        idx_in_pool = context["pool"].index(cand)
                        corr_val = context['obj_correlation'][key][idx_in_pool]
                    except (KeyError, ValueError):
                        # Skip if no data for this pair
                        continue
                    
                    neg_corr_values.append(max(0.0, -corr_val))  # Positive bonus only from negative correlations
            
            if len(neg_corr_values) > 0:
                corr_bonus = sum(neg_corr_values)/len(neg_corr_values)
        
        final_score = acq_value + (0.1 * corr_bonus)
        scores.append(final_score)

    return scores