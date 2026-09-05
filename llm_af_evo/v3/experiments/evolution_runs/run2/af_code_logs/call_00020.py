def score_pool(context):
    """Progress-adaptive exploitation-uncertainty blend with novelty boost during stagnation and hypervolume-normalized scores."""
    X_obs = context["X_obs"]
    names = context["objective_names"] 
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    campaign = context["campaign"]
    
    progress = campaign["progress"]
    stagnant_batches = campaign["stagnant_batches"]

    # Adaptive exploitation weight: start low, increase towards end
    w_exploit = 1.0 - max(0.0, min(1.0, (2 * progress) ** 3))
    w_uncertain = 1.0 - w_exploit
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize objectives to avoid scale bias
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        # Blend exploitation and uncertainty scores  
        score = w_exploit * (mu_sum / len(names)) + w_uncertain * sigma_norm

        # Add novelty bonus if stagnant
        if stagnation_batches > 0:
            distances = np.linalg.norm(X_obs - cand["x"], axis=1)
            nearest_distance = np.min(distances) 
            novel_bonus = max(0.0, 1e-6 / (nearest_distance + 1e-6))
            score += 0.5 * novel_bonus
            
        scores.append(score)

    return scores