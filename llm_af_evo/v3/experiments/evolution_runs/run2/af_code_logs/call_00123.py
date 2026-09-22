def score_pool(context):
    """Integrates predicted objective means with uncertainty-aware hypervolume contribution estimates using dynamic weighting and progress-adaptive scaling."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Use a sigmoidal blend for exploitation vs exploration
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.4)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)

        # Normalize by front range to avoid scale bias
        mu_norm = mu_sum / np.sum(front_range.values())
        sigma_norm = sigma_sum / np.sum(front_range.values())

        # Dynamic scaling of uncertainty based on progress and reference point distance  
        ref_distance = 1.0 - (mu_sum / np.sum(ref_point))
        scaled_sigma = sigma_norm * (1 + 3.0 * progress) * ref_distance

        score = w_exploit * mu_norm + (1 - w_exploit) * scaled_sigma
        scores.append(score)

    return scores