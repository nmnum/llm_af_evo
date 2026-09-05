def modifier(context):
    """Adaptive hypervolume expansion bonus based on candidate's predicted contribution to front diversity and campaign stagnation."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    y_max = np.max(context["Y_obs"], axis=0)
    ranges = y_max - ref_point
    normalized_ref = (ref_point - ref_point) / ranges  # Should be all zeros
    
    values = []
    for cand in context["pool"]:
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        norm_mean = (mean_vec - ref_point) / ranges
        hv_contribution = 1.0
        
        # Estimate how much this candidate would improve hypervolume if added to the front,
        # using a simplified "volume of dominated region" approach.
        cand_dominated_volume_ratio = np.prod(np.maximum(1e-8, (normalized_ref - norm_mean)))
        
        # Scale by acquisition value and stagnation level
        acq_value_norm = cand["acq_value_norm"]
        stagnant_factor = 0.5 * min(context["campaign"]["stagnant_batches"] / 3.0, 1.0)
        bonus_weight = max(0., (cand_dominated_volume_ratio - 0.2)) * stagnated_factor
        
        # Apply a dynamic weight that increases with campaign progress
        prog_weight = context["campaign"]["progress"]
        
        values.append(bonus_weight * acq_value_norm * prog_weight)

    return values