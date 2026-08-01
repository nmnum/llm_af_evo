def score_pool(context):
    """Pure exploitation: rank by predicted objective sum only."""
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        scores.append(gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"])
    return scores