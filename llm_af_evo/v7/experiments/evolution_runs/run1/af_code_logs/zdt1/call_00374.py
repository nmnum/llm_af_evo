def modifier(context):
    """Penalty for candidates that are likely dominated by existing points, based on GP posterior sampling and pairwise domination checks."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    n_samples = 20
    penalty_weight = 0.3
    
    # Sample from each candidate's posterior to estimate potential domination
    values = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        # Generate samples for this candidate across all objectives 
        means = [gp_posterior[name]["mean"] for name in names]
        stds = [gp_posterior[name]["std"] for name in names]

        sampled_points = np.random.normal(means, stds, (n_samples, len(names)))
        
        # Check if any of the samples are dominated by existing observations
        n_dominated_by_obs = 0
        
        y_min = np.array([context["ref_point_by_name"][name] for name in names])
        ranges = np.max(context["Y_obs"], axis=0) - y_min

        normalized_samples = (sampled_points - y_min) / ranges 
        normalized_front = (context["pareto_front"] - y_min) / ranges
        
        # Use a simple domination check: point is dominated if there exists another
        # with all objectives better or equal, and at least one strictly better.
        
        for sample in normalized_samples:
            dom_by_any_obs = False
            
            for front_point in normalized_front:
                if np.all(sample <= front_point) and not np.array_equal(sample, front_point):
                    dom_by_any_obs = True
                    break
                    
            n_dominated_by_obs += int(dom_by_any_obs)
            
        # Apply penalty based on fraction of samples that are dominated by current points,
        # scaled with campaign progress to reduce influence later.
        
        frac_dom = float(n_dominated_by_obs) / n_samples
        
        acq_value_norm = cand["acq_value_norm"]
                
        values.append(-penalty_weight * frac_dom * (1.0 - context["campaign"]["progress"]) * acq_value_norm)
    
    return values