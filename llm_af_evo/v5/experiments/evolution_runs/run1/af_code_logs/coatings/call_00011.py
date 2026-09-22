def score_pool(context):
    """Blend acquisition value with a progress-aware hypervolume bonus that rewards candidates expanding dominated regions more than those near or inside the current Pareto front."""
    names = context["objective_names"]
    ref_point = np.array([context['ref_point_by_name'][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted objective values
        pred_obj = [gp[name]["mean"] for name in names]
        
        # Compute hypervolume contribution using reference point and predicted objectives.
        hv_contribution = 1.0
        for i, obj_val in enumerate(pred_obj):
            if ref_point[i] > obj_val:
                hv_contribution *= (ref_point[i] - obj_val)
                
        acq_score = cand["acq_value_norm"]
        
        # Bonus based on how much the candidate expands a dominated region,
        # inversely weighted by proximity to current front.
        bonus_factor = 1.0
        if len(context['pareto_front']) > 0:
            min_dist_to_pf = float('inf')
            
            for pf_point in context["pareto_front"]:
                dist_sq = sum((pred_obj[i] - pf_point[i])**2 for i in range(len(names)))
                if dist_sq < min_dist_to_pf**2: 
                    min_dist_to_pf = np.sqrt(dist_sq)
                    
            # Normalize the distance to front by observed ranges
            norm_distance = 0.5 * (min_dist_to_pf / max(context["pareto_front_range"].values(), default=1))
            
            bonus_factor += max(0, 2 - norm_distance) ** 3
            
        scores.append(acq_score + 0.4 * hv_contribution * bonus_factor)
        
    return scores