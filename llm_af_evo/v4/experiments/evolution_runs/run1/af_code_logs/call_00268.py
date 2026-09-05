def score_pool(context):
    """Incentivize exploration near the periphery of current Pareto front by blending acquisition value with inverse distance to frontier edge."""
    names = context["objective_names"]
    
    # Compute reference point for hypervolume calculation (slightly outside pareto_front)
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp_mean = [cand["gp_posterior"][name]["mean"] for name in names]
        
        # Compute distance from candidate to the reference point (hypervolume expansion signal)
        dist_to_ref = np.linalg.norm(np.array(gp_mean) - ref_point)

        # Inverse of this gives preference to candidates expanding hypervolume more
        if dist_to_ref > 0:
            hv_signal = 1.0 / dist_to_ref 
        else:  
            hv_signal = float('inf')
            
        acq_value_scaled = cand["acq_value_norm"] * hv_signal
        
        scores.append(acq_value_scaled)
        
    return scores