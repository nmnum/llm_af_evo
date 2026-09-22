def score_pool(context):
    """Score candidates by acquisition value adjusted for progress-aware uncertainty and novelty."""
    names = context["objective_names"]
    
    # Use the provided normalized acquisition values directly 
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Normalize qualities (sum of means) to [0, 1]  
    qualities = []
    for cand in context["pool"]:
        mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names)
        qualities.append(mu_sum)

    q_min, q_max = min(qualities), max(qualities)
    
    if abs(q_max - q_min) < 1e-9:
        norm_qualities = np.array([0.5] * len(qualities))
    else: 
        norm_qualities = (np.array(qualities) - q_min) / (q_max - q_min)

    # Combine acquisition value and normalized quality
    combined_score = acq_values + 1.2 * norm_qualities
    
    # Adjust for progress using a sigmoid function that increases uncertainty weight as campaign progresses  
    p = context["campaign"]["progress"]
    
    # Early: low exploitation, high exploration (more UCB-like)
    # Late: higher exploitation
    ucb_weight = np.tanh(3. * (p - 0.5)) / 2 + 0.5
    
    scores = []
    for i in range(len(context["pool"])):
        gp_posterior = context["pool"][i]["gp_posterior"]
        
        # Compute normalized uncertainty 
        sigma_norm_sum = sum(gp_posterior[name]['std'] / (context['pareto_front_range'][name] + 1e-9)  
                             for name in names)
         
        ucb_term = ucb_weight * sigma_norm_sum
        
        final_score = combined_score[i]
        
        # Add uncertainty adjustment
        scores.append(final_score - 0.3*ucb_term)

    return scores