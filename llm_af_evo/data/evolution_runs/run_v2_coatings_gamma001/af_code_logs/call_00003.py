def score_pool(context):
    """Weight uncertainty term more heavily early in campaign, decaying over time, with extra novelty boost during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Decay factor for uncertainty term: higher early, lower later
    uncertainty_weight = 1.0 + (2.0 * progress)
    
    # Extra novelty boost when stagnant
    novelty_boost = 1.0 + min(stagnant_batches, 5) * 0.2
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        score = mu_sum + uncertainty_weight * sigma_norm * novelty_boost
        scores.append(score)
    return scores