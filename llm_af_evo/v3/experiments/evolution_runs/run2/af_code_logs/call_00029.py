def score_pool(context):
    """Greedy proximity-based suppression: suppress candidates that are too close to already selected ones, favoring diversity in candidate selection."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]

    # Use a fixed threshold for novelty suppression
    suppress_threshold = 0.1

    scores = []
    selected_x = []

    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        # Base score is exploitation plus uncertainty
        base_score = 0.5 * mu_sum + 1.0 * sigma_norm

        # Check proximity to already selected candidates (if any have been picked yet, or if we're early)
        cand_x = cand["x"]
        
        suppress_factor = 1.0
        
        for x_selected in selected_x:
            distance = np.linalg.norm(cand_x - x_selected) / max(front_range.values())
            
            # If candidate is too close to a previously picked one, reduce its score
            if distance < suppress_threshold and progress > 0.25:  
                suppress_factor *= (1e-6 + 1.0 - min(1.0, distance/suppress_threshold))
        
        final_score = base_score * suppress_factor
        
        scores.append(final_score)
    
    return scores