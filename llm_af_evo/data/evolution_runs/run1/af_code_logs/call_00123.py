def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        # Uncertainty term with progress-dependent weight
        uncertainty = (gp['Tm']['std']/front_range['Tm'] + 
                      gp['kD']['std']/front_range['kD'] + 
                      gp['viscosity']['std']/front_range['viscosity'])
        ucb_weight = 2.0 * (1.0 - progress)  # Exploit more as progress increases
        ucb_score = ucb_weight * uncertainty
        
        # Novelty term
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty_score = -dists.min()  # Negative because higher distance is better
        
        # Combine UCB and novelty
        scores.append(ucb_score + 0.5 * novelty_score)
    return scores