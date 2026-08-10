def score_pool(context):
    """Estimate pareto optimality probability per candidate using Monte Carlo sampled objectives, then rank by expected hypervolume improvement."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Sample from each candidate's posterior to estimate Pareto dominance
    n_samples = 100
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Draw samples from the joint GP distribution (simplified as independent)
        sampled_objectives = np.array([
            [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            for _ in range(n_samples)
        ])
        
        # Count how many times this candidate's sample dominates the current ref point
        n_dominates_ref = sum(
            all(sample_obj[i] >= ref_point[i] for i in range(len(names)))
            for sample_obj in sampled_objectives
        )
        
        # Probability of being Pareto-optimal (simplified estimate)
        pareto_prob = 1.0 - float(n_dominates_ref) / n_samples
        
        mu_sum = sum(gp[name]["mean"] for name in names)

        scores.append(mu_sum * pareto_prob)
    return scores