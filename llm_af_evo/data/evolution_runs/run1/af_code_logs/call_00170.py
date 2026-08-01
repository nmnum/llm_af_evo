def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        uncertainty = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        novelty = 1.0 / (1.0 + np.linalg.norm(X_obs - cand['x'], axis=1).min())
        exploitation_weight = max(0.3, 1.0 - progress * 0.7)
        scores.append(exploitation_weight * mu_sum + (1.0 - exploitation_weight) * uncertainty + novelty * 0.5)
    return scores