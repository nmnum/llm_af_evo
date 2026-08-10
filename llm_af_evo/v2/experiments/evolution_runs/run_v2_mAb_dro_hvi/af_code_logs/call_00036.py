def score_pool(context):
    """Exploitation with probabilistic pareto dominance estimation: rank by predicted objective sum adjusted for uncertainty-based Pareto probability."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        # Estimate the chance this candidate is Pareto-optimal by sampling its posterior
        n_samples = 100
        samples = np.random.normal(
            [gp[name]["mean"] for name in names],
            [gp[name]["std"] for name in names],
            (n_samples, len(names))
        )
        
        # Count how many sampled points are dominated by current Pareto front
        n_dominated = 0
        for sample in samples:
            is_dominated = False
            for pf_point in context["pareto_front"]:
                if all(sample[i] <= pf_point[i] + 1e-6 for i in range(len(names))):
                    # Point dominated by Pareto front, count it and move to next point 
                    n_dominated += 1  
                    is_dominated = True
                    break
        
        pareto_prob = (n_samples - n_dominated) / float(n_samples)
        
        scores.append(mu_sum * max(0.5, pareto_prob)) 
        
    return scores