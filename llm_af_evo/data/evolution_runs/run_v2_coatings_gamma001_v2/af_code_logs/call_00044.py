def score_pool(context):
    """Exploitation with dynamic uncertainty weighting, novelty encouragement during stagnation, and progress-based balancing."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Dynamic uncertainty weight: decreases as campaign progresses
    uncertainty_weight = max(0.2, 1.0 - progress * 0.7)
    
    # Novelty boost during stagnation, capped for stability
    novelty_factor = 1.0 + min(stagnant_batches, 5) * 0.15
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        score = mu_sum + uncertainty_weight * sigma_norm * novelty_factor
        scores.append(score)
    return scores