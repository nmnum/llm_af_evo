def score_pool(context):
    """Dynamic exploitation-uncertainty balance with stagnation-aware novelty boost and progressive exploration decay."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Decreasing uncertainty weight as progress advances
    uncertainty_weight = max(0.2, 1.0 - progress * 0.7)
    
    # Increase novelty bonus when stagnation is detected, up to a limit
    novelty_bonus = 1.0 + min(stagnant_batches, 4) * 0.15
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        score = mu_sum + uncertainty_weight * sigma_norm * novelty_bonus
        scores.append(score)
    return scores