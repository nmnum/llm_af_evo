def score_pool(context):
    """Score by predicted objective sum adjusted for uncertainty and penalized if too similar to top candidates."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Calculate base scores (muSum) and uncertainties (sigmaNorm)
    mu_sums, sigma_norms = [], []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        mu_sums.append(mu_sum)
        sigma_norms.append(sigma_norm)

    # Normalize scores to [0, 1]
    if len(mu_sums) > 0:
        max_mu = np.max(mu_sums)
        min_mu = np.min(mu_sums)
        range_mu = max(max_mu - min_mu, 1e-8)
        
        normalized_scores = [(mu_sum - min_mu)/range_mu for mu_sum in mu_sums]
    else:
        # Fallback if no candidates
        return [0.]*len(context["pool"])

    # Adjust scores by uncertainty (UCB-style) but with a twist: higher uncertainty reduces score,
    # unless it's near the Pareto front where we want to explore more.
    
    adjusted_scores = []
    for i, cand in enumerate(context["pool"]):
        gp = cand["gp_posterior"]
        
        mu_sum = mu_sums[i]
        sigma_norm = sigma_norms[i] 

        # Use a sigmoid-like decay on uncertainty effect
        ucb_factor = 1.0 / (1 + np.exp(-5 * (sigma_norm - 0.2))) 
        adjusted_score = normalized_scores[i]*ucb_factor

        # Penalize candidates that are too similar to already selected ones.
        penalized_score = adjusted_score
        
        for j in range(min(3, len(context["X_obs"]))):  
            if np.linalg.norm(cand['x'] - context["X_obs"][j]) < 0.1:
                penalized_score *= (1-0.2) # Reduce score by up to 20% 
                
        adjusted_scores.append(penalized_score)
        
    return adjusted_scores