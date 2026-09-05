def score_pool(context):
    """Estimate the probability that a candidate is Pareto optimal by sampling noisy objectives, then blend with acquisition value."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Use noise to estimate dominance probability (sample from GP posteriors)
    n_samples = 50
    pf = context["pareto_front"] 
    y_obs = context["Y_obs"]

    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample noisy predictions across objectives  
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean = gp_posterior[name]["mean"] 
            std = gp_posterior[name]["std"]
            samples[:,i] = np.random.normal(mean, std, n_samples)
        
        # Count how often this candidate dominates (or is dominated by) observed points
        cand_obj = [gp_posterior[n]['mean'] for n in names]
        dom_count = 0
        
        if len(pf) >= 3:
            # Compare against Pareto front  
            for pf_point in pf: 
                better_or_equal = np.all(samples <= pf_point)
                strictly_better = any(s > p for s, p in zip(cand_obj, pf_point))
                dom_count += (better_or_equal and strictly_better)  # candidate dominates PF point
        else:
            # Compare against all observations  
            for obs_y in y_obs: 
                better_or_equal = np.all(samples <= obs_y)
                strictly_better = any(s > o for s, o in zip(cand_obj, obs_y))
                dom_count += (better_or_equal and strictly_better)  # candidate dominates observation

        dominance_prob = min(1.0, max(0., float(dom_count)/n_samples)) 
        
        acq_score = cand["acq_value_norm"]
        
        # Blend acquisition value with estimated dominance probability
        score = acq_score + (dominance_prob * 0.3) 
        scores.append(score)
    
    return scores