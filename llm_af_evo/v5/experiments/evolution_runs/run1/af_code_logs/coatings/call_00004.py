def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a small uncertainty bonus."""
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        # Add a small uncertainty bonus: sum of normalized stds
        sigma_bonus = sum(cand["gp_posterior"][name]["std"] / context["pareto_front_range"][name]
                          for name in context["objective_names"])
        scores.append(acq + 0.1 * sigma_bonus)
    return scores