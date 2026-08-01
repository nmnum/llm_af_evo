def score_pool(context):
    """Combine exploitation with novelty and uncertainty."""
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation: weighted sum of means
        mu_sum = (gp["Tm"]["mean"] * 0.4 + 
                 gp["kD"]["mean"] * 0.3 + 
                 gp["viscosity"]["mean"] * 0.3)
        
        # Uncertainty: normalized standard deviation
        sigma_norm = (gp["Tm"]["std"] / front_range["Tm"] * 0.4 +
                     gp["kD"]["std"] / front_range["kD"] * 0.3 +
                     gp["viscosity"]["std"] / front_range["viscosity"] * 0.3)
        
        # Novelty: distance to nearest observed point
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = float(dists.min())
        
        # Balance exploitation, uncertainty, and novelty
        # Early: more weight on exploration (uncertainty + novelty)
        # Later: more weight on exploitation
        w_exploit = 0.3 + 0.7 * (1 - progress)
        w_uncert = 0.4 * (1 - progress)
        w_novelty = 0.3 * (1 - progress)
        
        score = (w_exploit * mu_sum + 
                w_uncert * sigma_norm + 
                w_novelty * novelty)
        
        scores.append(score)
    
    return scores