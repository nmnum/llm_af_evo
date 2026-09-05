def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    
    # Check if obj_correlation is provided and not empty
    if context["obj_correlation"]:
        corr_key_f1f2 = "f1,f2"
        
        # Extract the f1-f2 correlation values for each candidate in pool, default to 0.0 when key missing (shouldn't happen with valid input but just in case)
        correlations = np.array(context["obj_correlation"].get(corr_key_f1f2, [0.0] * len(context["pool"])))
        
        # Compute a bonus based on negative correlation
        avg_neg_corr_bonus = -np.mean(correlations[correlations < 0]) if any(c < 0 for c in correlations) else 0.0
        
    else:
        # No objective correlation signal available; contribute nothing to the score.
        avg_neg_corr_bonus = 0.0
    
    # Combine with acquisition value: dominant term is acq_value_norm, secondary bonus from negative correlations
    alpha = 1.0   # Weight for main acquisition term (no need to tune as it's already properly scaled)
    beta = 0.25   # Small weight for the correlation-based bonus
    
    for cand in context["pool"]:
        base_score = cand['acq_value_norm']
        
        score_with_bonus = alpha * base_score + beta * avg_neg_corr_bonus
        
        scores.append(score_with_bonus)

    return scores