def score_pool(context):
    """Adaptive exploitation and uncertainty bonus: dynamically balance mean and std based on campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Adaptively balance exploitation and exploration based on progress
        weight_exploitation = 1.0 - 0.5 * progress  # Decrease exploitation as campaign progresses
        weight_exploration = 0.5 + 0.5 * progress   # Increase exploration as campaign progresses
        scores.append(weight_exploitation * mu_sum + weight_exploration * sigma_norm)
    return scores