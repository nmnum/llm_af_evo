def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        # Exploitation + uncertainty tradeoff
        mu_sum = (gp['Tm']['mean'] + gp['kD']['mean'] + gp['viscosity']['mean'])
        sigma_norm = (gp['Tm']['std']/front_range['Tm'] + 
                      gp['kD']['std']/front_range['kD'] + 
                      gp['viscosity']['std']/front_range['viscosity'])
        # Blend exploitation and uncertainty based on progress
        w = 0.3 + 0.7 * progress  # Early: more uncertainty, late: more exploitation
        score = w * mu_sum + (1 - w) * sigma_norm
        # Add novelty bonus (distance to nearest observed point)
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = 1.0 / (1.0 + dists.min())  # Avoid division by zero
        score += 0.5 * novelty
        scores.append(score)
    return scores