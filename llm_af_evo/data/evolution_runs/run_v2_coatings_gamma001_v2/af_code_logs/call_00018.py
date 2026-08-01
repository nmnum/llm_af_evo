def score_pool(context):
    """Exploitation with dynamic uncertainty weighting, novelty encouragement during stagnation, and progressive exploration emphasis."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Start with strong exploitation, gradually add uncertainty and novelty
    exploitation_weight = 1.0 - min(progress * 1.5, 1.0)
    uncertainty_weight = max(0.2, 0.8 - progress * 0.6)
    novelty_factor = 1.0 + min(stagnant_batches, 5) * 0.1
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        score = exploitation_weight * mu_sum + uncertainty_weight * sigma_norm * novelty_factor
        scores.append(score)
    return scores