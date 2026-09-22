def score_pool(context):
    """Balances exploitation and uncertainty via adaptive weights tied to progress-dependent front density."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Density-aware blend: more weight on exploitation when the Pareto front is sparse (early stage)
    if len(context["pareto_front"]) > 1:
        frontier_density = np.mean(np.std(context["Y_obs"], axis=0))
        w_exploit_base = max(0.3, min(0.9, 1.0 - progress * 0.5)) # Decrease exploitation as front grows
    else:
        w_exploit_base = 0.8

    weights = np.array([w_exploit_base] + [1.0-w_exploit_base]*len(names))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized exploitation signal: mean of objectives
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty term scaled by progress and objective ranges, modulated dynamically  
        sigma_scaled = 0.5 * (1 + np.exp(-3*(progress-0.4))) * \
                       sum((gp[name]["std"]/front_range[name]) ** weights[i+1] for i,name in enumerate(names))

        score = w_exploit_base * mu_sum_norm + (1 - w_exploit_base) * sigma_scaled
        scores.append(score)
    
    return scores