def score_pool(context):
    """Balances exploitation and uncertainty via dynamic weighting derived from Pareto front geometry and ref point distance."""
    names = context["objective_names"]
    pareto_front = context["pareto_front"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Early exploration, later focus on promising regions
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.3)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation: normalized mean values 
        mu_norm_sum = sum(gp[name]["mean"] / max(1e-8, ref_point[i] - np.min(pareto_front[:, i]))  
                          for i, name in enumerate(names))
                
        # Uncertainty component scaled by progress and front geometry
        sigma_total = sum(gp[name]["std"] * (0.5 + 0.5 * progress) 
                          for name in names)
        
        score = w_exploit * mu_norm_sum + (1 - w_exploit) * sigma_total
        
        scores.append(score)

    return scores