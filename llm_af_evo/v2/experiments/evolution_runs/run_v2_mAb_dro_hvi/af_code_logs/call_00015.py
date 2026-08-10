def score_pool(context):
    """Exploitation with dynamic uncertainty weighting: higher predicted means plus adaptive UCB-style bonus based on campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    prog = context["campaign"]["progress"]
    
    # Dynamic weight for uncertainty bonus, decreasing as we explore more
    w_uncertainty = 1.0 - prog
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Combine exploitation and uncertainty with dynamic weight
        scores.append(mu_sum + w_uncertainty * sigma_norm)

    return scores