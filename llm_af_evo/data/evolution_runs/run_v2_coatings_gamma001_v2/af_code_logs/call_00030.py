def score_pool(context):
    """Exploitation with dynamic uncertainty bonus: sum of means plus a progress-adaptive uncertainty term to encourage exploration near the Pareto front."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Adapt uncertainty bonus based on progress: more exploration early, less later
        adaptive_weight = 0.5 * (1 - progress)
        scores.append(mu_sum + adaptive_weight * sigma_norm)
    return scores