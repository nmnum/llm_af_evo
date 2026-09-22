def score_pool(context):
    """Estimates probability that a candidate expands the pareto front by sampling noisy GP predictions and computing empirical domination."""
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    # Use 10 MC samples per candidate to estimate hypervolume expansion potential 
    n_samples = 10
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        
        # Sample noisy predictions from GP posteriors  
        samples = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            mean, std = gp[name]["mean"], gp[name]["std"]
            samples[:,i] = np.random.normal(mean, std, n_samples)
            
        # Compute how many times this candidate would dominate the current front
        domination_count = 0 
        pf = context["pareto_front"] 
        
        for sample in samples:
            is_dominant = True
            
            # Check if any point on PF dominates this sample  
            for pt in pf:                
                if all(sample[ax] <= pt[ax] + 1e-8 for ax in range(len(names))):
                    is_dominant = False
                    break
                    
            domination_count += int(is_dominant)
            
        # Score based on expected number of dominating points (normalized by sample count) 
        score = domination_count / n_samples  
        
        scores.append(score)

    return scores