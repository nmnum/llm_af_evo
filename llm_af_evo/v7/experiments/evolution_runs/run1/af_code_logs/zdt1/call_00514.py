def modifier(context):
    """Boost candidates that are predicted to dominate existing front points, with decay over campaign progress."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    front_range = np.array([context["pareto_front_range"][name] for name in names])
    
    values = []
    progress_factor = 1.0 - context["campaign"]["progress"] * 0.5
    
    # Normalize reference point and front range to [0, 1]
    ref_point_norm = (ref_point - np.min(context["Y_obs"], axis=0)) / front_range
    for cand in context["pool"]:
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Normalize candidate's predicted objectives to [0, 1] space using observed range 
        normalized_mean = (mean_vec - np.min(context["Y_obs"], axis=0)) / front_range
        
        # Compute hypervolume improvement estimate based on dominance
        dominates_front_points = sum(
            all(normalized_mean[i] >= ref_point_norm[i] for i in range(len(names))) and  
            any(normalized_mean[i] > ref_point_norm[i]) 
            for _ in context["pareto_front"]
        )
        
        # Apply a bonus if candidate is predicted to expand the dominated region
        hv_bonus = dominates_front_points * progress_factor
        
        values.append(hv_bonus)
    
    return values