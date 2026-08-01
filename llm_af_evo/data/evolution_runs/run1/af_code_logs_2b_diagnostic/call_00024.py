def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        uncertainty = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        novelty = np.linalg.norm(X_obs - cand['x'], axis=1).min()
        exploitation = (gp['Tm']['mean'] + gp['kD']['mean'] + gp['viscosity']['mean']) / 3.0
        scores.append(uncertainty * (2.0 + progress) + novelty * 0.1 + exploitation * (1.0 - progress))
    return scores