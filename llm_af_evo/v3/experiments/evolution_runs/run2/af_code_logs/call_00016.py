def score_pool(context):
    """Adaptive exploitation and uncertainty trade-off with progressive weighting based on campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    
    # Progressively shift from exploration to exploitation
    w_exploit = 1.0 - max(0.0, min(1.0, (2 * progress) ** 3))
    w_uncertain = 1.0 - w_exploit
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        # Blend exploitation and uncertainty with adaptive weights
        score = w_exploit * mu_sum + w_uncertain * sigma_norm
        scores.append(score)
    
    return scores