def score_pool(context):
    """Combines progress-aware exploitation with uncertainty-sensitive hypervolume difference estimates using dynamic reference point adjustment."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"].copy()
    progress = context["campaign"]["progress"]

    # Adjust reference point dynamically based on campaign progress to shift focus
    for i, name in enumerate(names):
        ref_point[i] += (1.0 - progress) * front_range[name]

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted means and uncertainties normalized by range  
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Hypervolume difference estimate using adjusted reference
        cand_values = np.array([gp[name]["mean"] for name in names])
        hv_diff = max(0.0, (ref_point[0] - cand_values[0]) * (ref_point[1] - cand_values[1]))

        score = mu_norm_sum + 2.0 * sigma_norm_sum + 0.5 * np.sqrt(hv_diff)
        
        scores.append(score)

    return scores