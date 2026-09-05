def score_pool(context):
    """Adapts uncertainty sensitivity based on progress and refines exploitation using normalized hypervolume gain estimates."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Progress-aware blending: early exploration, late exploitation
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation term based on normalized mean 
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty-aware component: scaled by progress to reduce impact over time
        sigma_total = sum(gp[name]["std"] for name in names) * (1.0 - 0.5 * progress)

        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * sigma_total
        
        scores.append(score)
    
    return scores