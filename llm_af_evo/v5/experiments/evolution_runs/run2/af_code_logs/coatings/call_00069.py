def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context["obj_correlation"]:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    
    for i, cand in enumerate(context["pool"]):
        acq_val = cand['acq_value_norm']
        
        bonus = 0.0
        if corr_keys:  
            neg_corr_count = 0
            total_neg_corr = 0.
            
            for key in corr_keys:
                val = context["obj_correlation"][key][i]
                
                name_a, name_b = key.split(',')
                idx_a = names.index(name_a)
                idx_b = names.index(name_b) 
                
                if (idx_a < len(context['pareto_front']) and
                    idx_b < len(context['pareto_front'])):
                    
                    front_range_a = context["pareto_front_range"][name_a]
                    front_range_b = context["pareto_front_range"][name_b]

                    # Check if the candidate's objectives are near current pareto fronts 
                    a_pred = cand["gp_posterior"][name_a]["mean"]
                    b_pred = cand["gp_posterior"][name_b]["mean"]

                    is_near_front_a = abs(a_pred - max(context['pareto_front'][:, idx_a])) < 0.1 * front_range_a
                    is_near_front_b = abs(b_pred - max(context['pareto_front'][:, idx_b])) < 0.1 * front_range_b

                    if val < 0 and (is_near_front_a or is_near_front_b):
                        total_neg_corr += min(1., abs(val))  
                        neg_corr_count += 1
                        
            bonus = total_neg_corr / max(neg_corr_count, 1) 

        scores.append(acq_val + 0.2 * bonus)
        
    return scores