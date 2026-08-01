def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        # Exploration: uncertainty normalized by front range
        uncertainty = (gp['Tm']['std']/front_range['Tm'] + 
                      gp['kD']['std']/front_range['kD'] + 
                      gp['viscosity']['std']/front_range['viscosity'])
        # Exploitation: mean values (already flipped to be higher-is-better)
        mean_val = (gp['Tm']['mean'] + gp['kD']['mean'] + gp['viscosity']['mean'])
        # Novelty: distance to nearest observed point
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = 1.0 / (dists.min() + 1e-8)  # Avoid division by zero
        
        # Weight exploration vs exploitation based on progress
        w_explore = 0.3 + 0.7 * progress  # Start with less exploration, increase over time
        w_novelty = 0.2 * (1 - progress)  # Less novelty weighting early
        
        score = w_explore * uncertainty + (1 - w_explore) * mean_val + w_novelty * novelty
        scores.append(score)
    return scores