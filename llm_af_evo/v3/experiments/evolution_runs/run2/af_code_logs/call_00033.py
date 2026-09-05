def score_pool(context):
    """Blend exploitation and uncertainty with a hypervolume-aware normalization and progressive exploration emphasis."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    progress = context["campaign"]["progress"]

    # Progressively shift from exploit-heavy to uncertain-weighted as campaign advances
    w_exploit = 0.7 * (1 - progress**2)
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names) 
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        # Apply progressive weighting
        score = w_exploit * mu_sum + (1 - w_exploit) * sigma_norm
        
        scores.append(score)
    
    return scores