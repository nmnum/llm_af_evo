def score_pool(context):
    """Exploitation with probabilistic Pareto dominance estimation: rank by expected hypervolume improvement adjusted for uncertainty."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Estimate probability of being Pareto optimal using Monte Carlo sampling
        n_samples = 100
        pareto_prob = 0.0
        
        for _ in range(n_samples):
            sample_objectives = np.array([np.random.normal(gp[name]["mean"], gp[name]["std"]) 
                                         for name in names])
            
            # If this candidate dominates the reference point, it's potentially Pareto optimal  
            if all(sample_objectives >= ref_point):
                pareto_prob += 1.0
                
        pareto_prob /= n_samples
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        scores.append(mu_sum * (1 + pareto_prob)) 
        
    return scores