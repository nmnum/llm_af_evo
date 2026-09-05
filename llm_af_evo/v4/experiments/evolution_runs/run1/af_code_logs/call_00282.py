def score_pool(context):
    """Incorporate uncertainty sensitivity into acquisition value using progress-aware exponential scaling."""
    scores = []
    prog = context["campaign"]["progress"]
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        sigma_sum = sum(cand["gp_posterior"][name]["std"] for name in context["objective_names"])
        # Exponentially scale uncertainty bonus based on campaign progress
        ucb_bonus = (1 - prog) * 0.5 * sigma_sum
        scores.append(acq + ucb_bonus)
    return scores