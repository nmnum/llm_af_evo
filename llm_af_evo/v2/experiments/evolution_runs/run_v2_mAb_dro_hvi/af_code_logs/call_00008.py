def score_pool(context):
    """Estimate Pareto-optimality probability via Monte Carlo sampling, then compute inverse-distance-weighted density among high-probability candidates."""
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
        
        # Count how many samples are not dominated by any point on the Pareto front
        n_pareto = 0 
        for sample in samples:
            is_dominated = False
            for frontier_point in front:
                if all(frontier_point[i] >= sample[i] for i in range(len(names))) and \
                   any(frontier_point[i] > sample[i] for i in range(len(names))):
                    is_dominated = True
                    break 
            if not is_dominated:  
                n_pareto += 1
        
        cand_probs.append(n_pareto / float(n_samples))
    
    # Compute inverse-distance-weighted density among candidates with high Pareto probability
    scores = []
    for i, (cand_x, prob) in enumerate(zip([c["x"] for c in context["pool"]], cand_probs)):
        if prob < threshold_prob:
            score = 0.0 
        else:  
            # Compute density using only candidates with high Pareto probability
            weights_sum = 0.
            for j, (other_x, other_prob) in enumerate(zip([c["x"] for c in context["pool"]], cand_probs)):
                if i != j and other_prob >= threshold_prob:
                    dist_sq = np.sum((cand_x - other_x)**2)
                    weight = 1. / max(dist_sq, 1e-8) 
                    weights_sum += weight
            score = prob * (weights_sum + 1.) # Add small constant to avoid zero density
    
        scores.append(score)

    return scores