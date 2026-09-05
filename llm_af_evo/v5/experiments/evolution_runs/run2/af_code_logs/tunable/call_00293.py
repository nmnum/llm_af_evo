def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    
    # Check if obj_correlation is populated (non-empty)
    has_corr_signal = bool(context.get('obj_correlation', {}))
    
    for cand in context["pool"]:
        acq_val_norm = cand['acq_value_norm']
        
        corr_bonus = 0.0
        if has_corr_signal:
            # Collect correlation values relevant to this candidate across all objective pairs
            total_neg_corr = 0.
            count_pairs = 0
            
            for key, vals in context["obj_correlation"].items():
                name_a, name_b = key.split(',')
                
                try: 
                    corr_val = vals[context["pool"].index(cand)] # index aligned with pool candidates
                except (IndexError, ValueError):
                    continue
                    
                if not isinstance(corr_val, float) or np.isnan(corr_val):  
                    continue

                total_neg_corr += max(0., -corr_val)
                count_pairs += 1
            
            corr_bonus = total_neg_corr / max(count_pairs, 1)

        # Combine acquisition value with correlation bonus (smaller weight to avoid overfitting)
        score = acq_val_norm + 0.2 * corr_bonus
        scores.append(score)
        
    return scores