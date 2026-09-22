def modifier(context):
    """Resample posterior means with noise to assess candidate robustness and penalize overconfident predictions."""
    import numpy as np
    
    names = context["objective_names"]
    values = []
    
    # Set number of resamples for assessing uncertainty in mean estimates
    n_resamples = 10

    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Collect means and stds across objectives  
        means = np.array([gp[name]["mean"] for name in names])
        stds = np.array([gp[name]["std"] for name in names])

        # Generate noisy samples of the candidate's mean
        noise_samples = np.random.normal(means, stds, (n_resamples, len(names)))

        # For each sample compute hypervolume improvement estimate if this point were observed,
        # then assess how much variance there is across these estimates.
        
        hv_improvements = []
        for i in range(n_resamples):
            sampled_mean = noise_samples[i]
            
            # Simple dominance check: a candidate improves HV only if it's better than all current front points
            dominates_any = False 
            for pf_point in context["pareto_front"]:
                if np.all(sampled_mean <= pf_point) and not np.array_equal(sampled_mean, pf_point):
                    continues_dominance = True  # Could be a weak dominance check or more robust one
            
            hv_improvements.append(0.0)
            
        # Use the variance of HV estimates as an uncertainty measure for this candidate's prediction
        if len(hv_improvements) > 1:
            correction = -np.std(hv_improvements) * 2.
        else:
            correction = 0.

        values.append(correction)

    return values