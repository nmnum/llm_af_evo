def score_pool(context):
    """Progress-aware hypervolume-based acquisition with uncertainty-weighted objective means and novelty penalty."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Normalize candidate objectives to [0, 1] range for consistency
    def normalize_objectives(cand):
        return {name: cand["gp_posterior"][name]["mean"] / front_range[name]
                for name in names}

    scores = []
    for i, cand in enumerate(context["pool"]):
        gp = cand["gp_posterior"]
        
        # Normalize the candidate's predicted objectives
        normalized_means = normalize_objectives(cand)
        
        # Compute hypervolume contribution using reference point (simplified estimate)  
        hv_contribution = 1.0 - np.prod([max(0, ref_point[j] - gp[names[j]]["mean"])
                                         for j in range(len(names))])
                
        # Blend exploitation and uncertainty with progress-dependent weights
        w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))
        
        mu_sum_norm = sum(normalized_means[name] for name in names)
        sigma_scaled = sum(gp[name]["std"] * (1.0 + progress) / front_range[name]
                           for name in names)

        # Combine hypervolume signal with exploitation and uncertainty terms
        score = w_exploit * hv_contribution + \
                (1 - w_exploit) * mu_sum_norm + \
                 0.5 * sigma_scaled

        scores.append(score)
    
    return scores