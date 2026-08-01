def score_pool(context):
    """Score by predicted objective sum plus a dynamic uncertainty bonus based on progress and front spread."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Dynamic weight: start with more exploration, shift towards exploitation
        w = 0.5 + 0.5 * (1 - progress)  # Decrease uncertainty weight as progress increases
        scores.append(mu_sum + w * sigma_norm)
    return scores