def score_pool(context):
    """Score by predicted objective sum plus a novelty-adjusted uncertainty bonus, balancing early exploration and late exploitation."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    
    # Early: more exploration; later: more exploitation
    ucb_weight = 3.0 * (1.0 - progress ** 0.5)
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        
        # Add a novelty term: inverse distance to nearest observed point
        cand_x = cand["x"]
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            novelty = 1.0 / (np.min(distances) + 1e-8)
        else:
            novelty = 1.0
        
        # Final score: exploitation + uncertainty bonus + novelty
        scores.append(mu_sum + ucb_weight * sigma_sum + 0.5 * novelty)
    
    return scores