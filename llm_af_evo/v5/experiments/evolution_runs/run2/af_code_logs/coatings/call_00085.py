def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    
    # Check if obj_correlation is provided and non-empty
    has_corr = bool(context.get("obj_correlation", {}))
    
    for cand in context["pool"]:
        acq_value = cand["acq_value_norm"]
        
        if not has_corr:
            score = acq_value
        else:
            # Compute correlation bonus only when correlations are available
            corr_bonus = 0.0
            
            obj_names = context["objective_names"]
            
            for i, name_a in enumerate(obj_names):
                for j, name_b in enumerate(obj_names[i+1:], start=i+1):
                    key = f"{name_a},{name_b}"
                    
                    # Check if the correlation data is present
                    corr_vals = context.get("obj_correlation", {}).get(key)
                    
                    if not corr_vals:
                        continue
                    
                    # Get this candidate's specific correlation value (index-aligned with pool)  
                    cand_corr_val = corr_vals[context["pool"].index(cand)]
                    
                    # Bonus: weight negative correlations more strongly
                    bonus_component = max(0.0, -cand_corr_val)
                    
                    corr_bonus += bonus_component
            
            # Normalize the bonus by number of objective pairs (or just add it as is if you prefer not to normalize for stability and simplicity in this context).
            
            score = acq_value + 0.1 * corr_bonus
        
        scores.append(score)

    return scores