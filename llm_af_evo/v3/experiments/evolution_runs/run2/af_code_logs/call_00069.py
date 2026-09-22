def score_pool(context):
    """Balances exploitation and uncertainty via progress-aware hypervolume difference estimation with dynamic scaling."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Use a smooth transition from exploration to exploitation based on campaign progress
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Estimate hypervolume improvement potential using predicted mean and std 
        mu_product = np.prod([gp[name]["mean"] for name in names])
        sigma_sum = sum(gp[name]["std"] for name in names)
        
        # Scale uncertainty contribution based on progress to encourage later exploitation
        adaptive_sigma_weight = 1.0 - w_exploit
        
        score = (w_exploit * mu_product + 
                adaptive_sigma_weight * sigma_sum / len(names))
                
        scores.append(score)

    return scores