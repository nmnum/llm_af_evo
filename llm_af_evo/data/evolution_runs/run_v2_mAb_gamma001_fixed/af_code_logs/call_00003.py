def score_pool(context):
    """Phase-aware weighted uncertainty plus exploitation, with decay over progress and novelty boost during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    campaign = context["campaign"]
    
    # Weight for uncertainty term that decays as budget is spent
    t = campaign["progress"]  # in [0,1]
    w_uncertainty = (1 - t) ** 2
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)

        # Combine exploitation and uncertainty with phase-aware weight
        score = mu_sum + w_uncertainty * sigma_norm
        
        scores.append(score)
        
    return scores