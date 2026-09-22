def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    
    # Check if obj_correlation is populated (non-empty dict)
    has_corr_signal = bool(context.get('obj_correlation', {}))
    
    for cand in context["pool"]:
        acq_value = cand['acq_value_norm']
        
        corr_bonus = 0.0
        if has_corr_signal:
            # Collect all correlation values relevant to this candidate (from obj_correlation)
            total_neg_corr = 0.
            count = 0
            
            for key, vals in context.get('obj_correlation', {}).items():
                try:
                    name_a, name_b = key.split(',')
                    
                    if cand['gp_posterior'].get(name_a) and cand['gp_posterior'].get(name_b):
                        # Get correlation value at this candidate index
                        corr_val = vals[context["pool"].index(cand)]
                        
                        total_neg_corr += max(0., -corr_val)
                        count += 1
                        
                except (ValueError, IndexError): 
                    continue
            
            if count > 0:
                avg_neg_correlation = total_neg_corr / float(count)  
                
                # Scale bonus to be small relative to acq_value
                corr_bonus = min(0.2 * acq_value, max(0., avg_neg_correlation))
        
        scores.append(acq_value + corr_bonus)
    
    return scores