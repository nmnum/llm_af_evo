def score_pool(context):
    """Estimates candidate dominance probability using GP posteriors sampled multiple times, favoring those more likely to be on Pareto front."""
    names = context["objective_names"]
    n_samples = 10
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        # Sample objectives from posterior
        samples = np.array([
            [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            for _ in range(n_samples)
        ])
        
        # Count how many times this candidate dominates (or is non-dominated)  
        dominating_count = 0
        total_comparisons = 0
        
        for sample in samples:
            is_dominating = True 
            for pf_point in context["pareto_front"]:
                if all(sample[objective] <= pf_point[i] for i, objective in enumerate(names)):
                    # This candidate's sampled point dominated by front
                    is_dominating = False  
                    break
                    
            if is_dominating:
                dominating_count += 1
                
        dominance_prob = dominating_count / n_samples 
        scores.append(dominance_prob)
        
    return scores