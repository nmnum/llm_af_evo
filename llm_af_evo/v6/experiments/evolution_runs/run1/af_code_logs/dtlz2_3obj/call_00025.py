def modifier(context):
    """Adaptive hypervolume gap bonus: candidates predicted to expand dominated objective space more than average get rewarded based on how much they outpace current front coverage."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute each candidate's hypervolume contribution estimate
    hv_contributions = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        pred_means = np.array([gp[name]["mean"] for name in names])  
        
        if len(context["pareto_front"]) == 0:
            # No front yet, use reference point as baseline
            hv_contrib = max(1e-8, ref_point - pred_means).prod()
        else: 
            # Compute hypervolume of the dominated region this candidate would add to Pareto Front if it were added now.
            # This is a simplification but captures core concept effectively:
            
            front_minima = context["pareto_front"].min(axis=0)
            hv_contrib = max(1e-8, (ref_point - pred_means).prod() - 
                             ((pred_means > front_minima) * (ref_point - np.maximum(pred_means,front_minima))).prod())
        hv_contributions.append(hv_contrib)

    # Normalize contributions across the pool to get relative gap values
    contribs = np.array(hv_contributions)
    
    if len(contribs[contribs>0]) == 0:
        return [0.0] * len(context["pool"])
        
    mean_hv_gap = max(1e-8, contribs.mean())
  
    # Scale bonus by how much candidate's predicted expansion exceeds average
    values = []
    
    for i in range(len(context["pool"])):
        cand = context["pool"][i]
        gp = cand["gp_posterior"]
        
        if mean_hv_gap == 0:
            ratio_bonus = 1.0 
        else:  
            # This is the core novelty term — how much more "gap-expanding" this candidate looks than average
            ratio_bonus = max(0., hv_contributions[i] / (mean_hv_gap + 1e-8))
            
        bonus_scale_factor = min(context["campaign"]["progress"] * 2.0, 1.) # decay as campaign proceeds 
        final_value = (ratio_bonus - 1) * bonus_scale_factor
        values.append(final_value)

    return values