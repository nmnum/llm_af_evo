def score_pool(context):
    """Estimate Pareto-optimality probability via MC sampling, then compute inverse-distance-weighted density among high-probability candidates."""
    n_samples = 25
    threshold_prob = 0.1
    names = context["objective_names"]
    front = context["pareto_front"]
    
    # Sample each candidate's GP posterior and estimate Pareto probability
    cand_probs = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        samples = np.array([[gp_posterior[name]["mean"] + gp_posterior[name]["std"] * np.random.normal()
                            for name in names] for _ in range(n_samples)])
        
        # Count how many samples are not dominated by any front point
        n_not_dominated = 0
        for sample in samples:
            is_dominated = False
            for q in front:
                if all(q[i] >= sample[i] for i in range(len(names))) and any(q[i] > sample[i] for i in range(len(names))):
                    is_dominated = True
                    break
            if not is_dominated:
                n_not_dominated += 1
        
        cand_probs.append(n_not_dominated / float(n_samples))
    
    # Compute inverse-distance-weighted density among high-probability candidates  
    scores = []
    for i, prob in enumerate(cand_probs):
        if prob < threshold_prob: 
            score = -np.inf
        else:
            x_i = context["pool"][i]["x"]
            total_weight = 0.0
            
            # Only consider other high-probability candidates  
            for j, (prob_j, cand_x) in enumerate(zip(cand_probs, [c['x'] for c in context["pool"]])):
                if prob_j < threshold_prob:
                    continue
                dist_sq = np.sum((x_i - cand_x)**2)
                weight = 1.0 / max(dist_sq, 1e-8) 
                total_weight += weight
                
            score = prob * (total_weight + 1e-6)

        scores.append(score)
        
    return scores