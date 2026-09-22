def score_pool(context):
    """Estimates candidate dominance likelihood using GP samples to guide exploration towards untested regions."""
    import numpy as np
    
    # Sample from each candidate's posterior to estimate Pareto membership probability
    n_samples = 50
    names = context["objective_names"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Draw samples from the joint distribution of objectives
        means = [gp[name]["mean"] for name in names]
        stds = [gp[name]["std"] for name in names]
        
        if len(stds) == 1:
            # For single objective, sample directly
            samples = np.random.normal(means[0], stds[0], n_samples)
        else:
            # Multivariate normal sampling (approximate with independent vars due to lack of covariances)
            samples_list = [np.random.normal(m, s, n_samples) for m,s in zip(means,stds)]
            samples = np.column_stack(samples_list)

        # Estimate how many sampled points are non-dominated by current front
        pf = context["pareto_front"]
        
        if len(pf) == 0:
            dominance_count = 0 
        else:    
            is_dominated = []
            for s in samples:
                dom = False
                for p in pf:
                    # Check domination (s dominates p)
                    if all(s[i] >= p[i] and not np.isclose(s[i],p[i]) for i in range(len(p))):
                        dom = True 
                        break  
                is_dominated.append(dom)

            dominance_count = sum(is_dominated) 

        # Score based on expected improvement potential
        score = 1.0 - (dominance_count / n_samples)
        
        scores.append(score)

    return scores