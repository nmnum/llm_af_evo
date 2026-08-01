def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        novelty = np.linalg.norm(X_obs - cand['x'], axis=1).min()
        uncertainty = (gp['Tm']['std']/front_range['Tm'] + 
                      gp['kD']['std']/front_range['kD'] + 
                      gp['viscosity']['std']/front_range['viscosity'])
        exploitation = (gp['Tm']['mean'] + gp['kD']['mean'] + gp['viscosity']['mean'])
        scores.append(0.5 * exploitation + 2.0 * uncertainty * max(0.0, 1.0 - progress) + 3.0 * novelty)
    return scores