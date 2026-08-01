def score_pool(context):
    """Adaptive exploitation-uncertainty tradeoff with dynamic weighting and stagnation-aware novelty bonus."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Dynamic uncertainty weight: higher early, lower later, but never below 0.5
    uncertainty_weight = max(0.5, 1.0 - progress * 0.7)
    
    # Stagnation-aware novelty bonus: stronger boost during stagnation
    novelty_factor = 1.0 + min(stagnant_batches, 4) * 0.25
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        score = mu_sum + uncertainty_weight * sigma_norm * novelty_factor
        scores.append(score)
    return scores