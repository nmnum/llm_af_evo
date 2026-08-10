def score_pool(context):
    """
    Estimate each candidate's potential contribution to hypervolume improvement by 
    resampling objective values from their GP posteriors and computing how often they 
    would dominate or be dominated in those samples — this captures both exploitation (high means)  
    and uncertainty-aware exploration without explicit novelty penalties.
    """    
    names = context["objective_names"]
    n_samples = 50
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]        
        # Sample predictions from the candidate's GP posteriors (for all objectives)
        samples = np.array([
            [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            for _ in range(n_samples)            
        ])
                
        dominated_count = 0
        non_dominated_count = 0
        
        # For each sample, check if it's dominant wrt current pareto front        
        for s in samples:
            is_dom_by_front = False
            for pf_point in context["pareto_front"]:
                if all(s[i] >= pf_point[i] for i in range(len(names))) and any(s[i] > pf_point[i] for i in range(len(names))):
                    # sample point dominates at least one front member strictly, so it's not dominated by the PF
                    break                    
                elif all(s[i] <= pf_point[i] for i in range(len(names))): 
                    is_dom_by_front = True            
            else:
                if not is_dom_by_front:  # s was non-dominated wrt current pareto front
                    non_dominated_count +=1
        score = (non_dominated_count / n_samples) * sum(gp[name]["mean"] for name in names)
        
        scores.append(score)

    return scores