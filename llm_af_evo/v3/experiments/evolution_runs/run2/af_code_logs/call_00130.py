def score_pool(context):
    """Exploits mean predictions with uncertainty-weighted progress scaling and normalized distance to Pareto front."""
    names = context["objective_names"]
    pareto_front = context["pareto_front"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize means by front range
        mu_norm_sum = sum(gp[name]["mean"] / (ref_point[i] - pareto_front.min(i)) 
                          for i, name in enumerate(names))
        
        # Scale uncertainty with progress and normalize  
        sigma_scaled_sum = sum(
            gp[name]["std"] * np.exp(progress) / (ref_point[i] - pareto_front.min(i)) 
            for i, name in enumerate(names)
        )
        
        score = mu_norm_sum + 0.5 * sigma_scaled_sum
        scores.append(score)

    return scores