def score_pool(context):
    """Estimate Pareto-optimality probability for each candidate via MC sampling; then compute inverse-distance-weighted density among high-probability points to guide trust-region selection."""
    import numpy as np
    
    n_samples = 25
    threshold_prob = 0.3
    names = context["objective_names"]
    
    # Sample from posteriors and calculate Pareto probabilities for each candidate once only
    pareto_probs = []
    samples_per_candidate = []

    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        samples = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"]
            
            # Sample from normal distribution
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
          
        samples_per_candidate.append(samples)

    for i, (cand, samples) in enumerate(zip(context["pool"], samples_per_candidate)):
        
        pareto_count = 0
        
        for sample_point in samples:
            
            is_pareto = True
            
            # Check if any existing front point dominates this sample
            for front_point in context['pareto_front']:
                dominated_by_any = all(front_point[j] >= sample_point[j] 
                                      for j in range(len(names)))
                
                strictly_dominated = any(front_point[j] > sample_point[j]
                                        for j in range(len(names)))

                if (dominated_by_any and strictly_dominated):
                    is_pareto = False
                    break
            
            # If no front point dominates this, it's Pareto-optimal so far  
            if is_pareto:
                pareto_count += 1
                
        prob = float(pareto_count) / n_samples 
        pareto_probs.append(prob)
    
    scores = []
        
    for i in range(len(context["pool"])):
      
        # Use already computed probability
        p_prob_i = pareto_probs[i]
                
        if (p_prob_i < threshold_prob):
            score = 0.0
            
        else:
            
            x_i = context['pool'][i]['x']
    
            weights_sum = 0.
            for j in range(len(context["pool"])):
                # Only consider candidates with high Pareto probability
                p_prob_j = pareto_probs[j]
                
                if (p_prob_j >= threshold_prob):
                    dist_squared = np.sum((x_i - context['pool'][j]['x']) ** 2)
                    
                    weight = 1.0 / max(1e-8, dist_squared) 
                    weights_sum += weight
                    
            score = p_prob_i * weights_sum
            
        scores.append(score)

    return scores