def score_pool(context):
    """Exploitation with dynamic uncertainty weighting and novelty encouragement during stagnation, adjusted for progress and Pareto front distance."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Dynamic uncertainty weight: decreases as campaign progresses
    uncertainty_weight = max(0.3, 1.0 - progress * 0.6)
    
    # Novelty boost during stagnation, capped for stability
    novelty_factor = 1.0 + min(stagnant_batches, 5) * 0.12
    
    # Distance from current Pareto front to encourage exploration
    pf_distances = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        cand_objectives = [gp[name]["mean"] for name in names]
        # Compute distance to the nearest point on the Pareto front
        min_dist = float('inf')
        for pf_point in context["pareto_front"]:
            dist = sum((cand_obj - pf_obj)**2 for cand_obj, pf_obj in zip(cand_objectives, pf_point))
            if dist < min_dist:
                min_dist = dist
        pf_distances.append(min_dist)
    
    # Normalize distances to [0, 1] range for scaling
    if len(pf_distances) > 0:
        max_dist = max(pf_distances)
        if max_dist > 0:
            normalized_distances = [d / max_dist for d in pf_distances]
        else:
            normalized_distances = [0.0] * len(pf_distances)
    else:
        normalized_distances = [0.0] * len(context["pool"])
    
    scores = []
    for i, cand in enumerate(context["pool"]):
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Incorporate distance from Pareto front into the score
        score = mu_sum + uncertainty_weight * sigma_norm * novelty_factor + 0.2 * normalized_distances[i]
        scores.append(score)
    return scores