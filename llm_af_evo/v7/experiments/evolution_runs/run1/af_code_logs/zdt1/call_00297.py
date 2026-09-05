def modifier(context):
    """Uncertainty-weighted distance from the current Pareto front weighted by acquisition strength."""
    
    import numpy as np
    
    names = context["objective_names"]
    if len(context.get("X_obs", [])) == 0:
        return [0.] * len(context["pool"])
        
    values = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        mean_vals = np.array([gp_posterior[name]["mean"] for name in names])
        std_vals = np.array([gp_posterior[name]["std"] for name in names])

        # Compute distance to Pareto front (minimum Euclidean distance)
        distances = []
        for pf_point in context["pareto_front"]:
            dist = np.linalg.norm(mean_vals - np.array(pf_point))
            distances.append(dist)
        
        min_distance = float(np.min(distances)) if distances else 0.0

        # Normalize by the range of each objective to make it scale-invariant
        front_range = np.array([context["pareto_front_range"][name] for name in names])
        normalized_dist = min_distance / (np.linalg.norm(front_range) + 1e-8)

        # Combine with acquisition strength: candidates that are both uncertain and have high acq_value_norm get bonus 
        ucb_weight = np.sum(std_vals)
        
        # Scale the correction based on how strong this candidate already is
        base_score = cand["acq_value_norm"]
        scaling_factor = 1.0 + (base_score * 2) 

        values.append(-normalized_dist * ucb_weight * scaling_factor)

    max_val, min_val = np.max(values), np.min(values)
    
    if abs(max_val - min_val) < 1e-8:
       return [val for val in values]
        
    # Normalize to prevent extremely large corrections
    normalized_values = []
    range_vals = (max_val - min_val + 1e-6 )
    for v in values:
        norm_v = float(v-min_val)/range_vals  
        normalized_values.append(norm_v * 0.5)
    
    return [v/2. for v in normalized_values]