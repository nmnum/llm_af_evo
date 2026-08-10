def score_pool(context):
    """Rank by predicted objective sum scaled by uncertainty-adjusted dominance likelihood."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted means and stds
        mu = [gp[name]["mean"] for name in names]
        sigma = np.array([gp[name]["std"] for name in names])

        # Estimate dominance likelihood via Monte Carlo sampling from the candidate's posterior.
        n_samples = 100
        samples = []
        
        # Sample each objective independently (assumes independent GP posteriors)
        for _ in range(n_samples):
            sample_mu = [gp[name]["mean"] + np.random.normal(0, gp[name]["std"]) 
                         for name in names]
            samples.append(sample_mu)

        samples_array = np.array(samples)  # shape: (n_samples, n_objectives)
        
        # Count how many of the sampled points are non-dominated by current front
        pf_points = context["pareto_front"]
        dominated_count = 0
        
        for sample in samples_array:
            is_dominated_by_any_pf_point = False
            
            for pf_point in pf_points: 
                if all(sample[i] <= pf_point[i] + 1e-8 for i in range(len(names))):
                    # All objectives are less than or equal to PF point (dominated)
                    dominated_count += 1
                    is_dominated_by_any_pf_point = True
                    break

            # If not yet determined as dominated, check if it's better along at least one objective,
            # and no worse in any other — i.e. non-dominated.
        
        dominance_likelihood = (n_samples - dominated_count) / n_samples
        
        mu_sum_scaled_by_uncertainty = sum(mu)
    
        scores.append(dominance_likelihood * mu_sum_scaled_by_uncertainty)

    return scores