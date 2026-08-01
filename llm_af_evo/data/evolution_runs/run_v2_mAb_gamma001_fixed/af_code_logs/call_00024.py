def score_pool(context):
    """Estimate improvement potential using hypervolume expansion minus uncertainty penalty."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Estimate how much each candidate would improve HV if observed,
    # then subtract a measure of prediction uncertainty (lower is better)
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Predicted means and stds
        mu_vals = [gp[name]["mean"] for name in names]
        
        # Hypervolume improvement estimate: simple additive margin from front,
        # scaled by the prediction uncertainty (higher mean, lower sigma is better)
        hv_improvement = sum(
            max(0.0, gp[name]["mean"] - context["ref_point_by_name"][name])
            for name in names
        )
        
        total_uncertainty = sum(gp[name]["std"] / front_range[name] for name in names)

        # Higher HV improvement with lower uncertainty is preferred.
        scores.append(hv_improvement * (1.0 + 0.5 * total_uncertainty))
    return scores