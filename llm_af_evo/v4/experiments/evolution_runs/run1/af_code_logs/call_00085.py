def score_pool(context):
    """Blend acquisition value with uncertainty-aware progress tracking to favor candidates that are both promising and sufficiently novel compared to the current front."""
    if not context["pool"]:
        return []
    
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    pf = context["pareto_front"] 
    pf_range = {name: context["pareto_front_range"][name] for name in names}
    
    # Normalize reference point to [0, 1]
    ref_norm = np.array([pf_range[name] if pf_range[name] > 0 else 1.0 for name in names])
    normalized_ref_point = (ref_point - context["pareto_front"].min(axis=0)) / ref_norm
    
    scores = []
    
    # Compute hypervolume contribution of each candidate's predicted objectives
    def hv_contribution(cand):
        cand_obj_pred = np.array([cand['gp_posterior'][name]['mean'] for name in names])
        
        if len(pf) == 0:
            return max(1e-8, (normalized_ref_point[0] - cand_obj_pred[0]) * 
                       (normalized_ref_point[1] - cand_obj_pred[1]))
            
        # Simple hypervolume calculation using reference point
        contrib = np.prod(np.maximum(normalized_ref_point - cand_obj_pred, 0))
        
        return max(1e-8, contrib)
    
    for i,cand in enumerate(context["pool"]):
        acq_value = cand['acq_value_norm']
        
        # Add a bonus based on how much the candidate improves over current front
        hv_improvement_bonus = hv_contribution(cand) 
        
        uncertainty_penalty = 0.1 * sum(
            (cand['gp_posterior'][name]['std'] / pf_range[name]) 
            for name in names if pf_range[name] > 0)
        
        # Final score: acquisition value plus improvement bonus, minus normalized uncertainty
        final_score = acq_value + hv_improvement_bonus - uncertainty_penalty
        
        scores.append(final_score)

    return scores