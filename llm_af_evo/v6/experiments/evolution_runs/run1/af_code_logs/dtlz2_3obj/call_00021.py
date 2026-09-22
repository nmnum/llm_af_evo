def modifier(context):
    """Add a front-expansion bonus that rewards candidates likely to extend the dominated hypervolume most if their predictions are correct."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        # Predicted mean vector
        pred_mean = np.array([gp[name]["mean"] for name in names])

        # Compute hypervolume contribution if this candidate were to be added
        # We estimate it by seeing how much more dominated space would become accessible  
        # with the addition of this point, assuming its prediction is accurate
        
        # The expansion potential relative to current pareto front (in objective-space)
        expanded_volume = 0.0 
        if len(context["pareto_front"]) > 0:
            pf_points = context["pareto_front"]
            
            for i in range(len(pf_points)):
                p = pf_points[i]
                
                # Candidate extends the dominated region from this point
                volume_contribution_if_added = np.prod(np.maximum(ref_point - pred_mean, 0)) \
                                              / (np.prod(np.maximum(ref_point - p, 0)) + 1e-8)
                    
                expanded_volume += max(0.0, volume_contribution_if_added)

        # Normalize by the range of each objective to make it scale-invariant
        front_range = np.array([context["pareto_front_range"][name] for name in names])
        
        if not (front_range == 0).any():
            normalized_expansion_potential = expanded_volume / (
                np.prod(front_range) + 1e-8)
            
             # Scale by acquisition strength so only promising candidates get the bonus
            values.append(cand["acq_value_norm"] * min(0.5, max(normalized_expansion_potential, 0)))
        else:
            values.append(0.)

    return values