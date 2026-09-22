def score_pool(context):
    """Resamples noisy Pareto front estimates to prioritize candidates that are likely to dominate current solutions."""
    names = context["objective_names"]
    pareto_front = context["pareto_front"] 
    ref_point = context["ref_point"]
    
    # Resample objectives from GP posteriors with noise to estimate dominance
    n_samples = 100  
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample noisy predictions for this candidate 
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean, std = gp[name]["mean"], gp[name]["std"]  
            samples[:,i] = np.random.normal(mean, std, n_samples)
            
        # Estimate how many times this sample would dominate current front points
        dominance_count = 0 
        for s in samples:
            dominates_any = any(
                all(s[i] >= pareto_front[j][i] for i in range(len(names))) and  
                any(s[i] > pareto_front[j][i] for i in range(len(names)))
                for j in range(len(pareto_front))
            )
            if dominates_any:
                dominance_count += 1
                
        # Score is proportion of samples that dominate
        score = dominance_count / n_samples 
        scores.append(score)
        
    return scores