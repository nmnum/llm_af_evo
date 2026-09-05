def score_pool(context):
    """Adapts between exploitation and uncertainty based on progress, with normalized scores to balance objective scales."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    campaign_progress = context["campaign"]["progress"]
    
    # Use a sigmoidal transition from exploration (uncertainty) to exploitation (mean)
    w_exploit = 0.3 + 0.7 * (1 - np.exp(-5 * campaign_progress))
    w_uncert = 1 - w_exploit
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize means and stds to objective ranges
        mu_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)

        score = w_exploit * mu_norm + w_uncert * sigma_norm
        
        scores.append(score)

    return scores