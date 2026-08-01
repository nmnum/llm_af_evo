def score_pool(context):
    """Adaptive exploitation and uncertainty bonus: dynamically balance means and uncertainty based on campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Adaptive weight: more exploitation early, more exploration later
        w_exploit = 1.0 - progress
        w_explore = progress
        scores.append(w_exploit * mu_sum + w_explore * sigma_norm)
    return scores