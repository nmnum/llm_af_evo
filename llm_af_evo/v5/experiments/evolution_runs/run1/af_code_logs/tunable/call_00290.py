def score_pool(context):
    """Score candidates by acquisition value adjusted for predictive entropy across objectives, encouraging exploration of under-covered objective dimensions."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Compute normalized standard deviations per objective
        sigmas_norm = [gp[name]["std"] / front_range[name] for name in names]
        
        # Entropy-like measure: sum of negative logs of std devs (higher entropy implies more uncertainty)
        entropies = [-np.log(sigma + 1e-9) for sigma in sigmas_norm]
        total_entropy = np.sum(entropies)

        # Base score is acquisition value, adjusted by the inverse of objective uncertainties
        base_score = cand["acq_value_norm"] * (1.0 + total_entropy)
        
        scores.append(base_score)
    
    return scores