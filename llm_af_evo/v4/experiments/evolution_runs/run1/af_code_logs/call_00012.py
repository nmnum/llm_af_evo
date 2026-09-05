def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a small uncertainty bonus."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        acquisition_value = cand["acq_value_norm"]
        scores.append(acquisition_value + 0.1 * sigma_norm) 
    return scores