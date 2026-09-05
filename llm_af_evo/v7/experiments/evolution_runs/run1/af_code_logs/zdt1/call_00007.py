def modifier(context):
    """Progress-decaying uncertainty bonus: rewards candidates the GP
    posterior is still uncertain about, more heavily early in the
    campaign. Returns ONLY this correction term — acq_value_norm is
    added automatically by the sandbox, not by this function."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    weight = 0.3376 * (1.0 - progress)
    terms = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_sum = sum(gp[name]["std"] / front_range[name] for name in names)
        terms.append(weight * sigma_sum)
    return terms