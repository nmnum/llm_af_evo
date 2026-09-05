def score_pool(context):
    """Estimates each candidate’s potential to expand hypervolume by sampling noisy predictions and measuring diversity."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    
    # Sample from the GP posteriors for uncertainty-aware HV estimation
    n_samples = 10
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Generate samples of objectives under posterior uncertainty (with noise)
        sampled_objectives = np.random.normal(
            [gp[name]["mean"] for name in names],
            [gp[name]["std"] for name in names]
        )

        # Estimate HV improvement using the ref point
        hv_improvement = 0.0
        
        if all(sampled_objectives > ref_point):
            # If sample dominates reference, compute hypervolume contribution directly (simplified)
            diff_f1 = sampled_objectives[0] - ref_point[0]
            diff_f2 = sampled_objectives[1] - ref_point[1]

            hv_improvement += max(0.0, diff_f1 * diff_f2)

        # Add a diversity bonus: penalize candidates near existing observations
        x_cand = cand["x"]
        
        distances_to_observed = np.linalg.norm(context['X_obs'] - x_cand, axis=1)
        min_distance = float(distances_to_observed.min())
        
        if not (min_distance < 0.05): # Avoid extreme penalization for close points
            diversity_bonus = max(0., 1 / (1 + np.exp(-2 * min_distance)))
            
            hv_improvement += diversity_bonus
        
        scores.append(hv_improvement)
    
    return scores