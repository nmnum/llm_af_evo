def score_pool(context):
    """Exploitation with progressive uncertainty weighting: blend mean prediction and normalized std based on campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    t = context["campaign"]["progress"]  # in [0,1]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)

        # Progressive weighting: start with exploitation, shift towards uncertainty as campaign progresses
        w_exploit = 1.0 - t * 0.5   # reduce exploit weight from 1 to 0.5 over time
        scores.append(w_exploit * mu_sum + (1 - w_exploit) * sigma_norm)
    return scores