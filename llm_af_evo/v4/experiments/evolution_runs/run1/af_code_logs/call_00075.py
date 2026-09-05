def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a small uncertainty bonus."""
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        ucb_bonus = 0.1145 * sum(cand["gp_posterior"][name]["std"] for name in context["objective_names"])
        scores.append(acq + ucb_bonus)
    return scores