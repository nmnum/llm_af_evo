def score_pool(context):
    """Exploitation-uncertainty balance with progressive uncertainty amplification and front-relative scoring."""
    names = context["objective_names"]
    pareto_front = context["pareto_front"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Amplify uncertainty more in early stages, less later
    ucb_weight = 2.0 * np.tanh(1.5 * (1 - progress))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Calculate hypervolume contribution relative to reference point and current front  
        mu_values = [gp[name]["mean"] for name in names]
        hv_contrib = max(0, ref_point[0] - min(mu_values[0], pareto_front[:, 0].min())) * \
                    max(0, ref_point[1] - min(mu_values[1], pareto_front[:, 1].min()))
        
        # Exploitation term: weighted sum of means
        exploitation = sum(gp[name]["mean"] for name in names)
        
        # Uncertainty term with dynamic scaling 
        uncertainty = ucb_weight * sum(gp[name]["std"] for name in names)

        score = hv_contrib + 0.5 * (exploitation - uncertainty)  
        scores.append(score)
    
    return scores