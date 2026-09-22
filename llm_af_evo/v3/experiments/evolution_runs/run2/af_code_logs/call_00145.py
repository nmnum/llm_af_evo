def score_pool(context):
    """Estimates Pareto-optimality probability per candidate via MC sampling, then scores based on local density of high-probability candidates."""
    import numpy as np
    
    n_samples = 25
    threshold_prob = 0.3
    names = context["objective_names"]
    
    # Step 1: Compute Pareto probabilities for each candidate using Monte Carlo samples.
    pareto_probs = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample from the joint GP posterior distribution (assuming independence)
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)

        # Check how many of these samples are not dominated by current Pareto front
        pf = context["pareto_front"]
        
        count_not_dominated = 0.0
        
        for sample in samples:
            
            is_dominateable = False
            
            for point in pf: 
                if all(point[i] >= sample[i] for i in range(len(names))) and any(
                    point[i] > sample[i] for i in range(len(names))
                ):
                    
                    # Sample dominated by this front point
                    is_dominateable = True
                    
                    break
            
            if not is_dominateable:
                
                count_not_dominated += 1.0
                
        pareto_probs.append(count_not_dominated / n_samples)
    
    # Step 2: Compute inverse-distance-weighted density for candidates with high Pareto probability.
    scores = []
    pf_points = np.array([cand["x"] for cand in context["pool"]])
    probs_array = np.array(pareto_probs)

    threshold_idxes = np.where(probs_array >= threshold_prob)[0]
    
    if len(threshold_idxes) == 0:
        # No points meet the probability criteria, fallback to just Pareto probabilities
        scores.extend(pareto_probs)
        
    else: 
                
        for i in range(len(context["pool"])):
            prob = pareto_probs[i] 
            
            local_density_score = 1.0
            
            if probs_array[i] >= threshold_prob:
                # Compute density of nearby high-probability candidates
                dists_to_highprob_points = []
            
                point_i_x = pf_points[i]
                
                for j in range(len(context["pool"])):
                    if i == j: continue
                    
                    prob_j = pareto_probs[j]

                    if probs_array[j] >= threshold_prob:
                        
                        distance_sqrd = np.sum((point_i_x - pf_points[j]) ** 2)
    
                        dists_to_highprob_points.append(1.0 / (distance_sqrd + 1e-8))
                
                local_density_score += sum(dists_to_highprob_points) if len(
                    dists_to_highprob_points
                ) > 0 else 0.
            
            scores.append(prob * local_density_score)
    
    return scores