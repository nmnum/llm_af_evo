def modifier(context):
    """Adaptive hypervolume gap bonus: rewards candidates that would extend the dominated region significantly more than current frontiers, scaled by how much exploration is needed based on stagnation."""
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    # Compute expected improvement in HV for each candidate
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Predicted objective vector (already flipped to maximize)
        pred_obj = np.array([gp[name]["mean"] for name in names])

        # Distance from reference point, scaled by front range
        ref_dist_scaled = np.sum((ref_point - pred_obj) / 
                                [context["pareto_front_range"][name] for name in names])
        
        values.append(ref_dist_scaled * (1.0 + 0.5 * context['campaign']['stagnant_batches']))

    # Normalize to [0, 0.3], scaled by max value across pool
    if len(values) > 0:
        norm_factor = np.max(values)
        if norm_factor != 0: 
            values = [(v / norm_factor) * 0.3 for v in values]
    
    return values