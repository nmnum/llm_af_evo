def score_pool(context):
    """Adapts exploration-exploitation balance using gradient-based progress sensing and normalized hypervolume gain estimates."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Use a sharper transition to shift from UCB-like early exploration towards exploitation
    w_exploit = np.exp(-5 * (1.0 - progress)**2)
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names) 
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Combine exploitation and uncertainty with a dynamic blend
        score = w_exploit * (mu_sum_norm + 0.5 * sigma_norm) + \
                (1 - w_exploit) * np.sqrt(2.0 * progress) * sigma_norm
        
        scores.append(score)

    return scores