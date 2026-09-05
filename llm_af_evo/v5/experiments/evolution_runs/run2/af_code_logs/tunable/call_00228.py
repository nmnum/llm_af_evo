def score_pool(context):
    """Estimate probability of a candidate improving front coverage by sampling noisy GP predictions and measuring how often it dominates existing points."""
    names = context["objective_names"]
    
    # Use acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Sample candidates' posteriors to estimate dominance probability
    n_samples = 20 
    pf = context["pareto_front"]

    if len(pf) == 0:
        dom_probs = np.zeros(len(context["pool"]))
    else:  
        dom_scores = []
        
        for cand in context["pool"]:
            gp_posterior = cand['gp_posterior']
            
            # Draw samples from GP posterior
            samples = [] 
            for _ in range(n_samples):
                sample_point = [np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"])  
                                for name in names]
                samples.append(sample_point)
                
            # Count how often sampled point dominates any front point            
            n_dominated = 0
            for samp in samples:
                dominated_by_front = False 
                for pf_pt in pf:                    
                    if all(samp[i] >= pf_pt[i] and (samp[i] > pf_pt[i]) for i in range(len(names))):
                        # Samp dominates this front point  
                        dominated_by_front = True
                        break
                        
                if not dominated_by_front:
                    n_dominated += 1
                    
            dom_prob = float(n_dominated) / n_samples 
            dom_scores.append(dom_prob)
            
        dom_probs = np.array(dom_scores)

    # Blend acquisition score with dominance probability (slightly favor acq)
    scores = acq_scores + 0.3 * dom_probs
    return list(scores)