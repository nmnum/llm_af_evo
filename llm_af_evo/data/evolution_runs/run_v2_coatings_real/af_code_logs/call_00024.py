def score_pool(context):
    """Exploitation-weighted uncertainty bonus: blend of predicted mean and uncertainty, tuned by progress."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Weight exploitation vs exploration based on progress: early = more explore, late = more exploit
        w_exploit = 0.3 + 0.7 * (1 - progress)  # Decrease exploitation weight as campaign progresses
        scores.append(w_exploit * mu_sum + (1 - w_exploit) * sigma_sum)
    return scores