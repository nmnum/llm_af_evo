def score_pool(context):
    """Score by predicted objective sum adjusted for uncertainty and penalized based on proximity to existing observations."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute base scores as mean of objectives (already oriented correctly)
    mu_sums = np.array([sum(cand["gp_posterior"][name]["mean"] for name in names) 
                        for cand in context["pool"]])
        
    # Normalize uncertainties relative to front range
    sigmas_normed = np.array([
        sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)
        for cand in context["pool"]
    ])
    
    # UCB-style score: exploit mean, penalize uncertainty  
    ucb_scores = mu_sums - 0.5 * sigmas_normed
    
    if len(context["X_obs"]) == 0:
        return list(ucb_scores)

    X_pool = np.array([cand["x"] for cand in context["pool"]])
    
    # Compute distances from each candidate to all previously observed points
    dists_to_observed = np.min(np.sum((X_pool[:, None, :] - context["X_obs"][None, :, :]) ** 2, axis=2), axis=1)
        
    # Scale distance penalty: candidates near existing observations get lower scores  
    min_dist_penalty = 0.5
    max_dist_penalty = 3.0
    
    dist_penalties = np.clip(dists_to_observed / (np.max(dists_to_observed) + 1e-9), 
                             min_dist_penalty, max_dist_penalty)
    
    # Apply distance penalty to UCB scores  
    final_scores = ucb_scores - dist_penalties

    return list(final_scores)