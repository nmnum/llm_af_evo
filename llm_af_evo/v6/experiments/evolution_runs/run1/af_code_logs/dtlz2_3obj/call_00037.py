def modifier(context):
    """Adaptive hypervolume expansion bonus: reward candidates that would significantly increase the dominated volume if they were Pareto-optimal."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute how much each candidate could expand HV if it was on the front
    values = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        # Estimate mean objectives (already flipped to maximize)
        means = np.array([gp_posterior[name]["mean"] for name in names])
        
        # Compute hypervolume contribution if this candidate were non-dominated
        hv_contribution = 1.0
        for i, obj_mean in enumerate(means):
            ref_val = ref_point[i]
            if obj_mean > ref_val: 
                continue   # This objective is already worse than reference (i.e., outside the feasible region)
            
            # Contribution to HV from this dimension with candidate's mean as boundary  
            hv_contribution *= (ref_val - obj_mean)

        values.append(hv_contribution * 0.1) 

    return np.array(values)