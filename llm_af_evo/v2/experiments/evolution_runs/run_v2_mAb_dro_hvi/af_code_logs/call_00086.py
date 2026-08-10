def score_pool(context):
    """Exploitation-aware uncertainty scoring with dynamic resampling-based pareto dominance probability."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Estimate the Pareto-optimal probability by sampling from each candidate's posterior
    n_samples = 100
    pareto_probs = []
    scores = []

    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample predictions across objectives 
        samples = np.array([
            [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            for _ in range(n_samples)
        ])
                
        # Count how many sampled points are not dominated by current pareto front
        n_non_dominated = 0        
        for sample_point in samples:
            is_pareto = True            
            for pf_point in context["pareto_front"]:
                if all(sample_point[i] <= pf_point[i] + 1e-8 for i in range(len(names))):
                    # Sample point dominated by a PF point
                    is_pareto = False 
                    break
                    
            n_non_dominated += int(is_pareto)
            
        pareto_prob = float(n_non_dominated) / n_samples        
        mu_sum = sum(gp[name]["mean"] for name in names)
                
        # Adjust score based on probability of being Pareto optimal, with uncertainty penalty
        scores.append(mu_sum * (1.0 + 2.5*pareto_prob - pareto_prob**2))
    
    return scores