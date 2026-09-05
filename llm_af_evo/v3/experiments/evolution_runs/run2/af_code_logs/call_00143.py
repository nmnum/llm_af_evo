def score_pool(context):
    """Estimates Pareto optimality probability for each candidate via Monte Carlo sampling,
       then scores candidates based on local density of high-probability points to encourage
       trust-region-like exploration."""
    
    n_samples = 25
    threshold_prob = 0.3
    
    # Sample from GP posteriors and compute Pareto probabilities
    pareto_probs = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        samples = np.zeros((n_samples, len(context["objective_names"])))
        for i, name in enumerate(context["objective_names"]):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"]
            # Sample from normal distribution (already flipped to maximize)
            samples[:, i] = np.random.normal(mean_val, std_val, n_samples)

        front_points = context['pareto_front']
        
        count_pareto = 0
        for sample in samples:
            is_dominated_by_any = False
            # Check if any point from current pareto dominates this sample
            for q_point in front_points: 
                dominated_all = True  
                strictly_better_exists = False
                
                for j, obj_val in enumerate(sample):
                    if not (q_point[j] >= obj_val):  # If the frontier doesn't dominate it on one dimension...
                        dominated_all = False
                        break
                        
                if dominated_all:
                    better_in_some_dim = any(q_point[i] > sample[i] 
                                            for i in range(len(context["objective_names"])))
                    
                    if better_in_some_dim:  
                        is_dominated_by_any = True
                        break
            
            # If no point from front dominates this, it counts as Pareto-optimal-so-far
            if not is_dominated_by_any:
                count_pareto += 1
                
        prob = float(count_pareto) / n_samples 
        pareto_probs.append(prob)

    scores = []
    
    for i in range(len(context["pool"])):
        
        # Use inverse-distance-weighted density of high-probability candidates
        cand_x = context['pool'][i]['x']
        numerator_sum = 0.0  
        denominator_sum = 1e-8
        
        prob_i = pareto_probs[i]
                
        for j in range(len(context["pool"])):
            if i == j: continue
            
            # Check whether the candidate at index `j` is sufficiently high probability
            prob_j = pareto_probs[j] 
                        
            dist_squared = np.sum((cand_x - context['pool'][j]['x']) ** 2)
            
            weight_factor = (1.0 / max(dist_squared, 1e-6)) if prob_j >= threshold_prob else 0
            
            numerator_sum += weight_factor * prob_j
            denominator_sum += weight_factor

        density_score = numerator_sum / denominator_sum
        
        # Final score: prioritize high probability and dense regions of such points  
        scores.append(density_score)

    return scores