def score_pool(context):
    """Score candidates based on predicted objective sum adjusted by uncertainty and novelty distance to observed points."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    X_obs = context["X_obs"]
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Novelty bonus: inverse distance to nearest observed point
        x_cand = cand["x"]
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - x_cand, axis=1)
            novelty_bonus = 1.0 / (np.min(distances) + 1e-8)
        else:
            novelty_bonus = 1.0
        
        scores.append(mu_sum + sigma_norm * novelty_bonus)

    return scores