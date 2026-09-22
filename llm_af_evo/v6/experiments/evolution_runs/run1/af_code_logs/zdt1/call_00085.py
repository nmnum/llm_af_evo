def modifier(context):
    """Add a dynamic uncertainty bonus that encourages exploration of under-covered regions in objective space by rewarding candidates with high predicted variance and low dominance potential."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute the mean GP standard deviation across objectives for each candidate
    sigma_norms = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_sum = sum(gp[name]["std"] / front_range[name] for name in names)
        sigma_norms.append(sigma_sum)

    # Compute the mean objective value (as a proxy for dominance potential) 
    mu_means = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_mean = sum(gp[name]["mean"] / front_range[name] for name in names)
        mu_means.append(mu_mean)

    # Apply correction: higher uncertainty, lower mean -> larger bonus
    weight = 0.3 * (1 - context["campaign"]["progress"])
    values = []
    
    for i, cand in enumerate(context["pool"]):
        sigma_norm = sigma_norms[i]
        mu_mean = mu_means[i] 
        # Bonus increases with uncertainty and decreases with mean objective value
        bonus = weight * sigma_norm * (1 - mu_mean)
        values.append(bonus)

    return values