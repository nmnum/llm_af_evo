def score_pool(context):
    """weighted sum of: novelty_min(3.22), sigma_sum_norm_early(0.96), novelty_stagnation(4.68)"""
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.2195) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (0.9607) * ((gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity']) * max(0.0, 1.0 - progress))
        s = s + (4.6819) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        scores.append(s)
    return scores
