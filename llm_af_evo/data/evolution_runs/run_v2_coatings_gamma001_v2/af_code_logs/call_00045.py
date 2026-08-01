def score_pool(context):
    """Exploitation with dynamic uncertainty weighting, novelty encouragement during stagnation, and progress-based balancing, plus hypervolume improvement estimate for better long-term guidance."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    pareto_front = context["pareto_front"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Dynamic uncertainty weight: decreases as campaign progresses
    uncertainty_weight = max(0.3, 1.0 - progress * 0.6)
    
    # Novelty boost during stagnation, capped for stability
    novelty_factor = 1.0 + min(stagnant_batches, 5) * 0.12
    
    # Estimate hypervolume improvement based on candidate's position relative to Pareto front
    hv_improvement_scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu = [gp[name]["mean"] for name in names]
        
        # Simple HV estimate: distance from reference point to candidate's predicted objectives
        hv_improvement = 1.0
        for i, (obj_val, ref_val) in enumerate(zip(mu, ref_point)):
            if obj_val > ref_val:
                hv_improvement *= (obj_val - ref_val) / front_range[names[i]]
        hv_improvement_scores.append(hv_improvement)
    
    scores = []
    for i, cand in enumerate(context["pool"]):
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        score = mu_sum + uncertainty_weight * sigma_norm * novelty_factor + hv_improvement_scores[i]
        scores.append(score)
    return scores