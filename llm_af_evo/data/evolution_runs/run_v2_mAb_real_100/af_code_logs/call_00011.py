def score_pool(context):
    """Exploitation with progressive uncertainty weighting and novelty penalty."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    
    # Progressive weight: start with exploitation, shift toward uncertainty as progress increases
    w_exploit = 1.0 - min(progress * 2.0, 1.0)
    w_uncertainty = min(progress * 2.0, 1.0)
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Combine exploitation and uncertainty with progressive weights
        score = w_exploit * mu_sum + w_uncertainty * sigma_norm
        
        # Add novelty penalty: candidates closer to observed points get lower scores
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - cand["x"], axis=1)
            min_distance = np.min(distances)
            # Inverse of distance as a proxy for novelty (higher = more novel)
            novelty = 1.0 / (1.0 + min_distance)
            score = score * (1.0 + 0.5 * novelty)  # Scale novelty impact
            
        scores.append(score)
    
    return scores