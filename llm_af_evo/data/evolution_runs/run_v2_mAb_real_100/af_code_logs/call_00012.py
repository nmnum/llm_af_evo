def score_pool(context):
    """Exploitation with uncertainty-weighted novelty: blend predicted means and normalized uncertainty, adjusted by campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Early progress: balance exploitation and exploration; later: exploit more
        w_exploit = 0.5 + 0.5 * (1 - progress)
        scores.append(w_exploit * mu_sum + (1 - w_exploit) * sigma_norm)
    return scores