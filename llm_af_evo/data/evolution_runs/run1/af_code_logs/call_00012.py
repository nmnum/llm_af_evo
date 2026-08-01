def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.3861) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (3.2712) * (gp['Tm']['mean'] + gp['kD']['mean'] + gp['viscosity']['mean'])
        scores.append(s)
    return scores
