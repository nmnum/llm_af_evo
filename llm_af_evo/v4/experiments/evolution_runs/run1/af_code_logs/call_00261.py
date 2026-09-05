def score_pool(context):
    """Blend hypervolume improvement with a correlation-based bonus when objective correlations are available."""
    scores = []
    
    # Check if obj_correlation is populated (not empty)
    if context["obj_correlation"]:
        corr_key_f1f2 = "f1,f2"
        
        # Extract the f1-f2 correlation values for each candidate
        corrs = np.array(context["obj_correlation"][corr_key_f1f2])
        
        # Compute a bonus based on negative correlations (penalize positive, reward negative)
        corr_bonus = -np.mean(corrs)  # Negative average gives higher bonuses when most are negatively correlated
        
    else:
        # No correlation signal available; contribute zero to all candidates
        corr_bonus = 0.0

    for cand in context["pool"]:
        acq_value_norm = cand['acq_value_norm']
        
        # Combine acquisition value with the bonus term (smaller weight)
        score = acq_value_norm + 0.1 * corr_bonus
        
        scores.append(score)

    return scores