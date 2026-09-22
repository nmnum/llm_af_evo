def score_pool(context):
    """Adaptively balance acquisition value and uncertainty using progress-aware weights to shift from exploration to exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    prog = context["campaign"]["progress"]

    # Progress-adaptive weight: early=more uncertain, late=more acq
    w_acq = 0.3 + 0.7 * (1 - prog)
    w_uncertainty = 1 - w_acq

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized uncertainty term 
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)

        acq = cand["acq_value_norm"]

        score = w_acq * acq + w_uncertainty * sigma_norm
        scores.append(score)
    return scores