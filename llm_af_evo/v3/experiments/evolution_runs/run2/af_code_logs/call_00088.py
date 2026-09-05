def score_pool(context):
    """Integrates normalized mean objectives with an exponentially decaying uncertainty signal and progressive hypervolume-aware blending."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Blend exploitation and exploration using a smooth sigmoidal transition
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.4)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Exponentially decaying uncertainty term
        sigma_decay = np.exp(-3 * progress) 
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * sigma_norm * sigma_decay
        
        scores.append(score)
    
    return scores