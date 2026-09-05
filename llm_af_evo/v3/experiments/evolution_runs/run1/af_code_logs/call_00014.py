def score_pool(context):
    """Adaptive UCB with novelty bonus from objective space distances, balancing exploitation and exploration."""
    y_obs = context["Y_obs"]
    names = context["objective_names"] 
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]

    # Adaptive weight: more exploitation early, more exploration later
    w_exploit = 0.3 + 0.7 * progress

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        
        # UCB-style score based on normalized mean and std 
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)  
        ucb_score = w_exploit * mu_sum + (1.0 - w_exploit) * sigma_norm

        # Objective space novelty: inverse distance to nearest observed point
        gp_mean = np.array([gp[name]["mean"] for name in names])
        if len(y_obs) > 0:
            dists = np.linalg.norm(y_obs - gp_mean, axis=1)
            min_dist = np.min(dists)
            novelty_bonus = 3.674 / (min_dist + 1e-8)
        else:
            # No observations yet: high bonus to encourage exploration
            novelty_bonus = 3.674

        scores.append(ucb_score + novelty_bonus)

    return scores