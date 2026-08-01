def score_pool(context):
    """Exploitation with dynamic uncertainty bonus: sum of means plus a scaled uncertainty term that increases as progress is made and stagnation occurs."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    campaign = context["campaign"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Increase exploration bonus as campaign progresses and if stagnating
        progress_factor = 1.0 + campaign["progress"] * 0.5 + campaign["stagnant_batches"] * 0.3
        scores.append(mu_sum + progress_factor * 0.5 * sigma_norm)
    return scores