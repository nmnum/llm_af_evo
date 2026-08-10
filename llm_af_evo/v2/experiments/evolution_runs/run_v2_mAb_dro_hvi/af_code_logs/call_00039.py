def score_pool(context):
    """Exploitation with uncertainty-aware ranking: sum of means plus scaled std penalty, adjusted by progress-driven exploration weight."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    
    # Dynamically adjust the balance between exploitation and exploration
    exploit_weight = 1.0 - max(0., (progress * 2.0) - 1.) ** 2
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Blend exploitation and exploration based on progress
        scores.append(mu_sum - exploit_weight * 0.5 * sigma_norm)

    return scores