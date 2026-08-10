def score_pool(context):
    """Blend UCB-style exploration with novelty bonus; boost uncertainty early and add stagnation-based diversity."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Dynamic weight for uncertainty: higher early, lower later
    w_uncertainty = 1.5 * (1 - progress)
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # UCB-style score with dynamic uncertainty weight
        ucb_score = mu_sum + w_uncertainty * sigma_norm
        
        # Add novelty bonus if stagnating, or when close to end of campaign (to encourage diversity at the end)  
        if stagnant_batches >= 2:
            cand_x = cand["x"]
            X_obs = context["X_obs"] 
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            min_distance = np.min(distances)
            novelty_bonus = 1.0 / (min_distance + 1e-9)  
            
        elif progress > 0.75:
            # Encourage diversity near the end of campaign
            cand_x = cand["x"]
            X_obs = context["X_obs"] 
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            min_distance = np.min(distances)  
            
        else:  novelty_bonus = 0.0
            
        scores.append(ucb_score + 0.5 * novelty_bonus)

    return scores