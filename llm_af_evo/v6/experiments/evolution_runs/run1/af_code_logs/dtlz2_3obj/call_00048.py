def modifier(context):
    """Penalty for candidates with low uncertainty-adjusted acquisition value, encouraging exploitation of confident predictions."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Compute normalized mean and std for each objective
        means_norm = np.array([gp[name]["mean"] / front_range[name] for name in names])
        stds_norm = np.array([gp[name]["std"] / front_range[name] for name in names])

        # Calculate uncertainty-adjusted acquisition value (higher is better)
        ucb_value = np.sum(means_norm) - 0.5 * np.mean(stds_norm)

        # Use the raw acq_value_norm as baseline and add penalty if UCB score low
        base_acq = cand["acq_value_norm"]
        
        # Apply a multiplicative factor to scale correction based on how confident we are in this candidate's value estimate.
        confidence_factor = 1.0 - np.mean(stds_norm) / (np.max([gp[name]["std"] for name in names]) + 1e-8)
        
        penalty = max(0., ucb_value * base_acq * confidence_factor)

        values.append(-penalty) 

    return values