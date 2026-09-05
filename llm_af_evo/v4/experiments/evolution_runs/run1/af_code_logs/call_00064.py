def score_pool(context):
    """Estimate each candidate's potential to expand hypervolume by sampling noisy posterior draws and penalize candidates that are too similar to already selected ones."""
    names = context["objective_names"]
    
    # Use acquisition value as base, but adjust based on uncertainty and novelty
    scores = []
    X_obs = context["X_obs"] 
    pool_size = len(context["pool"])
    
    if len(X_obs) == 0:
        # No observations yet: simple acq_value_norm score with small UCB bonus  
        for cand in context["pool"]:
            ucb_bonus = sum(cand["gp_posterior"][name]["std"] for name in names)
            scores.append(cand["acq_value_norm"] + 0.1 * ucb_bonus) 
    else:
        # Compute distances from each candidate to all previously observed points
        cand_xs = np.array([cand['x'] for cand in context["pool"]])
        
        # Use a noise-aware approach: compute how much improvement is expected,
        # but suppress candidates that are too similar (repulsion)
        repulse_threshold = 0.1
        
        base_scores = [cand["acq_value_norm"] for cand in context["pool"]]
            
        distances_to_observed = np.array([
            min(np.linalg.norm(cand_x - x_obs) 
                for x_obs in X_obs) if len(X_obs) > 0 else float('inf')
            for cand_x in cand_xs
        ])
        
        # Simple repulsion: reduce score of candidates that are too close to existing ones  
        penalty = np.zeros(pool_size)
        for i, dist in enumerate(distances_to_observed):
            if dist < repulse_threshold:
                penalty[i] -= (repulse_threshold - dist) * 2.0
                
        # Add small UCB bonus
        ucb_bonus_terms = [
            sum(cand["gp_posterior"][name]["std"] for name in names)
            for cand in context["pool"]
        ]
        
        final_scores = np.array(base_scores) + \
                       0.1 * np.array(ucb_bonus_terms) + penalty
        
        scores.extend(final_scores.tolist())
    
    return scores