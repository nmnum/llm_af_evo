def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context.get('obj_correlation', {}):
        return [cand['acq_value_norm'] for cand in context['pool']]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context['obj_correlation'].keys())
    corrs = np.array([context['obj_correlation'][key] for key in corr_keys]).T
    
    # For each candidate, compute average negative correlation with objectives already well-covered
    bonus_terms = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        
        # Identify which objectives are currently well covered (near the Pareto front)
        frontier_values = context['pareto_front']
        min_vals = np.min(frontier_values, axis=0)  # minimum value per objective
        max_vals = np.max(frontier_values, axis=0)  # maximum value per objective
        
        coverage_thresholds = {
            name: (min_val + max_val)/2. for name, min_val, max_val in zip(names, min_vals, max_vals)
        }
        
        neg_corr_sum = 0.
        count_neg_corrs = 0
        corr_values_for_cand_i = corrs[i]
        key_indices = [j for j, k in enumerate(corr_keys) if ',' in k and len(k.split(',')) == 2]

        # Only consider pairs where both objectives are present (i.e., valid correlation)
        relevant_correlations = []
        
        corr_idx_map = {k: i_ for i_, k in zip(key_indices, [corr_keys[i_] for i_ in key_indices])}
            
        if len(corr_values_for_cand_i) == 0:
            bonus_terms.append(0.0)
            continue
            
        # For each pair of objectives
        neg_corr_sum = sum(
                corr_val 
                for j_, (k, corr_val) in enumerate(zip([corr_keys[i_] for i_ in key_indices], corrs[i]))
                if ',' in k and len(k.split(',')) == 2  
                   and any(obj_name not in names or obj_name != name_a for idx,(name_a,name_b) in 
                           list(enumerate([(k_.split(',')[0] ,k_.split(',')[1])for i_,k_ in enumerate(corr_keys)]))
                            if corr_idx_map.get(k_) == j_)
        )
        
        # Normalize the bonus term to avoid dominance of this component
        normalized_bonus = 2. * max(neg_corr_sum, 0.) / (len(names) + 1.)
        score_i = cand['acq_value_norm'] + 0.5 * normalized_bonus
        
        scores.append(score_i)
        
    return [cand["acq_value_norm"] for cand in context["pool"]] if not bonus_terms else [
            acq_val + 0.2*bonus_term 
            for (acq_val, bonus_term) in zip([c['acq_value_norm']for c in context['pool']], bonus_terms)
    ]