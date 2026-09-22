def score_pool(context):
    """Integrates progressive exploitation weight with uncertainty-adjusted hypervolume gain to balance exploration and adaptation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Adaptive exploit/Explore trade-off: S-shaped curve centered at 0.5
    w_exploit = np.exp(-3 * (1 - progress)**2) / (np.exp(4*(1-progress)) + np.exp(4*progress))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized mean contribution to hypervolume
        mu_norm_sum = sum(gp[name]["mean"] * front_range[name] / (ref_point[i] - min(front_range[name], 1e-6)) 
                          for i, name in enumerate(names))
                
        # Uncertainty scaled by progress and squared for stronger early penalization  
        sigma_total = np.sqrt(sum((gp[name]["std"])**2 for name in names) * (0.5 + 0.5*progress)**2)

        score = w_exploit * mu_norm_sum - (1 - w_exploit) * sigma_total
        
        scores.append(score)
    
    return scores