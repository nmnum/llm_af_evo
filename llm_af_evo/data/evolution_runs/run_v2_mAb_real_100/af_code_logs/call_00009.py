def score_pool(context):
    """Exploitation with uncertainty-weighted bonus: sum of means plus a bonus for predicted improvement over current front, scaled by uncertainty."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        # Estimate how much this candidate would improve hypervolume
        # by computing the volume between ref_point and the candidate's prediction
        # if it were to dominate the current front
        hv_contribution = 1.0
        for name in names:
            pred = gp[name]["mean"]
            # If candidate is better than ref point, it contributes to HV
            # Otherwise, it doesn't help (we use 0 contribution)
            if pred > ref_point[names.index(name)]:
                hv_contribution *= (pred - ref_point[names.index(name)])
            else:
                hv_contribution = 0.0
                break
        # Add a bonus based on how much better this candidate is than the current front,
        # weighted by its uncertainty (lower uncertainty = higher confidence in the bonus)
        bonus = 0.0
        if hv_contribution > 0:
            # Compute the normalized margin to the front for each objective
            margins = [max(0, gp[name]["mean"] - max(pf[names.index(name)] for pf in context["pareto_front"])) 
                       for name in names]
            # Weight bonus by inverse of uncertainty (lower std = higher confidence)
            total_std = sum(gp[name]["std"] for name in names)
            if total_std > 0:
                bonus = sum(margin / front_range[name] * gp[name]["std"] / total_std 
                            for margin, name in zip(margins, names))
        scores.append(mu_sum + 0.5 * bonus)
    return scores