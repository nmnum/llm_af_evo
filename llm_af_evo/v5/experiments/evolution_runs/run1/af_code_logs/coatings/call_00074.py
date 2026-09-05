def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    scores = []
    
    # Check if obj_correlation is populated (non-empty dict)
    if context.get('obj_correlation', {}):  # If not empty, process the signal
        
        names = context["objective_names"]
        
        # Compute average negative correlation for each candidate
        bonus_scores = np.zeros(len(context['pool']))
        
        num_pairs = 0

        # Iterate over all pairs of objectives (e.g., 'obj_1,obj_2')
        corr_keys = [k for k in context.get('obj_correlation', {}).keys() if ',' in k]
    
        for key in corr_keys:
            parts = key.split(',')
            name_a, name_b = parts[0], parts[1]

            # Check that both objectives are present
            if name_a not in names or name_b not in names: continue

            corrs = context['obj_correlation'][key]  # List of correlations per candidate
            
            for i, corr_val in enumerate(corrs):
                bonus_scores[i] += max(0.0, -corr_val)   # Only count negative correlation
                num_pairs +=1
                
        if num_pairs > 0:
            bonus_scores /= float(num_pairs)
    else: 
         bonus_scores = np.zeros(len(context['pool'])) 

    
     # Final score is dominated by acq_value_norm but includes a small boost from correlations  
    for i, cand in enumerate(context["pool"]):
       base_score = cand["acq_value_norm"]
       scores.append(base_score + 0.1 * bonus_scores[i])

        
    return scores