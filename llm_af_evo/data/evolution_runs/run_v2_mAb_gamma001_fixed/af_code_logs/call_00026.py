def score_pool(context):
    """Score by predicted objective sum adjusted for uncertainty and penalized diversity based on proximity to top candidates."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute base scores as mean plus normalized std (UCB-style)
    ucb_scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        ucb_scores.append(mu_sum + 2.0 * sigma_norm)

    # Identify top candidates (e.g., top 1/3 by UCB score) to define a reference set
    sorted_indices = np.argsort(ucb_scores)[::-1]
    n_top = max(1, len(context["pool"]) // 3)
    top_candidate_indices = sorted_indices[:n_top]

    # Compute diversity penalty: for each candidate, measure inverse similarity 
    # with the best candidates in terms of their feature space proximity
    X_pool = np.array([cand["x"] for cand in context["pool"]])
    
    # Distance matrix between all points (excluding self)
    dists_sq = np.sum((X_pool[:, None, :] - X_pool[None, :, :]) ** 2, axis=2) 
    np.fill_diagonal(dists_sq, np.inf)

    diversity_penalties = []
    for i in range(len(context["pool"])):
        # Find nearest top candidates (excluding self)
        distances_to_top = dists_sq[i][top_candidate_indices]
        
        if len(distances_to_top) > 0:
            min_dist = np.min(distances_to_top)
            
            # Inverse of distance as penalty, scaled to [0.5, 1] so no candidate is fully penalized
            penalty_factor = max(0.5, 2 * (min_dist / dists_sq[i].max() if dists_sq[i].max() > 0 else 1))
        else:
            # No top candidates found; minimal impact of diversity 
            penalty_factor = 1

        diversity_penalties.append(penalty_factor)

    scores = [ucb_scores[i] * diversity_penalties[i] for i in range(len(context["pool"]))]
    
    return scores