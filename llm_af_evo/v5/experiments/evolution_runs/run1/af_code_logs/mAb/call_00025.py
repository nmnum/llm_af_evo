def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    
    # Check if obj_correlation is provided and not empty
    has_corr_signal = bool(context.get('obj_correlation', {}))
    
    for cand in context["pool"]:
        acq_val_norm = cand['acq_value_norm']
        
        corr_bonus = 0.0
        
        if has_corr_signal:
            # Compute average negative correlation with objectives already well-covered
            total_neg_corr = 0.
            count_pairs = 0
            
            gp_posterior = cand["gp_posterior"]
            
            for key, correlations in context['obj_correlation'].items():
                name_a, name_b = key.split(',')
                
                if (name_a not in gp_posterior or 
                    name_b not in gp_posterior):
                    continue
                
                # Use the correlation value associated with this candidate
                corr_val = correlations[context["pool"].index(cand)]
                
                # Only consider negative correlations as bonus signal
                if corr_val < 0:
                    total_neg_corr += -corr_val  
                    count_pairs += 1
            
            avg_neg_corr = (total_neg_corr / max(count_pairs, 1))
            
            # Weight the correlation-based boost to be secondary 
            corr_bonus = 0.2 * avg_neg_corr
        
        scores.append(acq_val_norm + corr_bonus)
    
    return scores