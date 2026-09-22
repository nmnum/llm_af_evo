def modifier(context):
    """Adaptive uncertainty bonus scaled by how close candidates are to the current Pareto front."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    ref_point = np.array(context["ref_point"])
    
    if len(pf) < 1:
        return [0.0] * len(context["pool"])

    # Compute reference point adjusted for each objective's range
    front_range = {name: context["pareto_front_range"][name] 
                   for name in names}
    
    values = []
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distance to the nearest point on Pareto front
        diffs = pf - gp_mean[None, :]
        dists_sq = np.sum(diffs**2, axis=1)
        min_dist_sq = np.min(dists_sq) if len(dists_sq) > 0 else float('inf')
        
        # Normalize distance by the range of objectives to make it scale-invariant
        normalized_distance_squared = (
            (min_dist_sq / sum(front_range[name]**2 for name in names)) 
            if any(front_range.values()) and min_dist_sq != float('inf')  
            else 1.0)
        
        # Compute uncertainty bonus: higher when farther from front, scaled by how uncertain the candidate is
        sigma_norm = np.mean([cand["gp_posterior"][name]["std"] for name in names])
        
        weight_factor = (normalized_distance_squared ** -2) if normalized_distance_squared > 0 else float('inf')
        bonus = min(1.5 * weight_factor * sigma_norm, 0.8)
    
        values.append(bonus)

    return values