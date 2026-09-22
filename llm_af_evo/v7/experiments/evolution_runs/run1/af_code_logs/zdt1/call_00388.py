def modifier(context):
    """Bonus for candidates that are likely to be pareto optimal, estimated by sampling noisy GP predictions and computing empirical domination counts."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    n_samples = 50
    values = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample from the joint GP posterior to estimate probability of being Pareto optimal
        means = np.array([gp_posterior[name]["mean"] for name in names])
        stds = np.array([gp_posterior[name]["std"] for name in names])

        samples = []
        for _ in range(n_samples):
            sample_means = means + np.random.randn(len(names)) * stds
            samples.append(sample_means)

        # Count how many sampled points are dominated by observed ones (i.e., not Pareto optimal)
        n_dominated_by_observed = 0
        y_min = [context["ref_point_by_name"][name] for name in names]
        
        for sample_mean in samples:
            is_pareto = True
            
            # Check if this point dominates any observed points (inverted dominance check: not dominated by obs)
            for i, row in enumerate(context["Y_obs"]):
                dom = all(row[j] >= sample_mean[j] - 1e-8 for j in range(len(names))) and \
                      any(row[j] > sample_mean[j] + 1e-8 for j in range(len(names)))
                
                if not dom:
                    continue
                
                # If this point is dominated by an observed one, it's NOT pareto optimal
                n_dominated_by_observed += 1
                is_pareto = False
            
            # For the purpose of our bonus logic we want to encourage candidates that are likely 
            # *not* already known to be non-Pareto (i.e., unlikely dominated by current observations)
        
        prob_not_dom_by_current = max(0.0, 1 - n_dominated_by_observed / float(n_samples))
    
        acq_value_norm = cand["acq_value_norm"]
        # Scale bonus based on acquisition value and how likely candidate is to be Pareto optimal
        scaled_bonus = min(prob_not_dom_by_current * 2., 1.) * (0.5 + 0.3*min(acq_value_norm, 1)) 

        values.append(scaled_bonus)

    return values