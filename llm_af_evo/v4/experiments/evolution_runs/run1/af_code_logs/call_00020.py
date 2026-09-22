def score_pool(context):
    """Estimate hypervolume expansion potential by resampling noisy objectives and penalizing candidates dominated in most samples."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    # Set up the number of Monte Carlo samples for estimating dominance
    n_samples = 100

    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        # Draw noisy samples from each objective's posterior distribution (mean and std)
        f_values = np.array([
            np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"], n_samples) 
            for name in names
        ]).T  # shape: [n_samples, num_objectives]

        # Compute hypervolume contribution of this candidate across all samples,
        # i.e., how much better the dominated region would be if we had perfect info.
        
        hv_contributions = []
        for sample_f_vals in f_values:
            # If any component is worse than reference point (i.e. not contributing to HV),
            # that's fine — hypervolume computation handles it automatically by using min of
            # the objective and ref_point value, but we can also directly filter.
            
            if all(sample_f_vals[i] <= ref_point[i] for i in range(len(names))):
                hv_contributions.append(0.0)
            else:
                contrib = np.prod([max(ref_point[i], sample_f_vals[i]) - min(ref_point[i], sample_f_vals[i])
                                   for i in range(len(names)) if not (sample_f_vals[i] <= ref_point[i])])
                # Only include the contribution that actually improves dominated space
                hv_contributions.append(contrib)

        expected_hv_improvement = np.mean(hv_contributions)
        
        scores.append(expected_hv_improvement + 0.1 * cand["acq_value_norm"])

    return scores