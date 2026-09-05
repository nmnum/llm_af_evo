def score_pool(context):
    """Adapts exploration-exploitation balance using uncertainty-aware hypervolume signal with dynamic scaling and progress-driven blending."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Blend exploitation weight: start low, increase gradually towards mid-campaign
    w_exploit = 0.2 + 0.6 * (1 - np.exp(-3 * progress))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        # Dynamic scaling of uncertainty based on progress and current exploitation weight
        scaled_sigma = np.sqrt(sigma_norm * (1.0 + 3.0 * w_exploit))

        score = w_exploit * mu_sum_norm + (1 - w_exploit) * scaled_sigma
        
        scores.append(score)
    
    return scores