def score_pool(context):
    """Score candidates by combining acquisition value with an entropy-based diversity bonus that penalizes overconfidence in dominated regions."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute base scores using acq_value_norm as dominant signal
    base_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Estimate entropy-based diversity bonus from GP posteriors (uncertainty normalized by front range)
    uncertainty_bonus = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_sum = sum(gp[name]["std"] / front_range[name] for name in names if front_range[name] > 0)
        # Normalize to [0,1], then invert so high entropy (wide uncertainty) gets higher bonus
        normalized_entropy = min(1.0, max(0.0, sigma_sum))
        diversity_bonus = 1 - normalized_entropy  
        uncertainty_bonus.append(diversity_bonus)

    scores = base_scores + np.array(uncertainty_bonus) * 0.3
    
    # Normalize final scores to [0, 1] range for consistent ranking
    if len(scores) > 1:
        score_range = max(scores) - min(scores)
        if score_range != 0: 
            scores = (scores - min(scores)) / score_range

    return list(scores)