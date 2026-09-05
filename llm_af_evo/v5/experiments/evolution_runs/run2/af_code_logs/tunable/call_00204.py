def score_pool(context):
    """Estimates Pareto-optimality probability for each candidate via Monte Carlo sampling, then scores candidates based on a density of high-probability points in their neighborhood."""
    
    import numpy as np
    
    n_samples = 25
    threshold_prob = 0.3
    names = context["objective_names"]
    front = context["pareto_front"]
    X_pool = np.array([cand['x'] for cand in context['pool']])
    
    # Compute Pareto probabilities once per candidate
    pareto_probs = []
    for i, cand in enumerate(context['pool']):
        gp_posterior = cand['gp_posterior']
        
        samples = []
        for _ in range(n_samples):
            sample_obj_vals = [np.random.normal(gp_posterior[name]["mean"], 
                                                max(1e-6, gp_posterior[name]["std"]))  
                               for name in names]
            samples.append(sample_obj_vals)
            
        n_pareto = 0
        for s in samples:
            dominated_by_front = False
            for q in front:
                if all(q[i] >= s[i] for i in range(len(s))) and any(q[i] > s[i] for i in range(len(s))):
                    dominated_by_front = True
                    break
            
            if not dominated_by_front:
                n_pareto += 1
                
        prob = float(n_pareto) / len(samples)
        pareto_probs.append(prob)

    # Compute density score based on high-probability neighbors only, inverse-distance weighted 
    scores = []
    
    for i in range(len(context['pool'])):  
        
        if pareto_probs[i] < threshold_prob:
            score = 0.0
        else:
            
            distances = np.linalg.norm(X_pool - X_pool[i], axis=1)
            
            # Weight by inverse of distance (avoiding division by zero) 
            weights = []
            for d in distances:  
                if d == 0.0:
                    w = float('inf')   # Self-weight is infinity, so always included
                else:
                    w = 1. / max(d, 1e-8)
                    
                weights.append(w)

            
            total_weighted_prob = sum(p * w for p,w in zip(pareto_probs,weights) if p >= threshold_prob )
    
            # Normalize by the number of neighbors with high probability 
            n_neighbors_highprob = len([p for p in pareto_probs if p >= threshold_prob])
        
            score = total_weighted_prob / max(1.0,n_neighbors_highprob)
            
        scores.append(score)

    return scores