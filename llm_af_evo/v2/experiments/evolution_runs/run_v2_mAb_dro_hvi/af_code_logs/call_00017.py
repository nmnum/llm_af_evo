def score_pool(context):
    """Adaptive exploitation and uncertainty tradeoff: dynamically balance mean profit against normalized standard deviation based on campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Adaptive weight: early on, favour exploration (higher uncertainty bonus);
        # later, lean more towards exploitation.
        progress = context["campaign"]["progress"]
        w_explore = 1.0 - progress
        scores.append(mu_sum + w_explore * sigma_norm)
    return scores