def score_pool(context):
    """Blend acquisition value with an adaptive uncertainty-reward that grows as front becomes dense and stagnation increases."""
    names = context["objective_names"]
    ref_point = context["ref_point_by_name"]
    scores = []
    
    # Compute how much the current Pareto front spans objective space
    if len(context["pareto_front"]) >= 2:
        front_spread = np.array([
            max(context["pareto_front"][:, i]) - min(context["pareto_front"][:, i])
            for i in range(len(names))
        ])
        # Normalize by reference point to get a relative density measure
        rel_density = np.mean(front_spread / list(ref_point.values()))
    else:
        rel_density = 1.0
    
    stagnation_factor = min(1., context["campaign"]["stagnant_batches"] * 0.2)
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Base score from acquisition value
        base_score = cand["acq_value_norm"]

        # Dynamic uncertainty bonus: increase when front is dense and stagnation high 
        sigma_sum = sum(gp[name]["std"] for name in names)
        normalized_sigma = sigma_sum / (rel_density + 1e-9) * (stagnation_factor + 0.5)

        scores.append(base_score + 0.3 * normalized_sigma)
    
    return scores