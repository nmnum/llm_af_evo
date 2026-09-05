def score_pool(context):
    """Blend acquisition value with an entropy-based uncertainty signal that dynamically adjusts exploration intensity based on Pareto front sparsity and campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute a dynamic entropy penalty term
    n_pf = len(context['pareto_front'])
    if n_pf < 2:
        entropies = [0.0]*len(context["pool"])
    else:
        front_means = np.mean(context['pareto_front'], axis=0)
        distances_to_mean = []
        for cand in context["pool"]:
            gp_posterior = cand["gp_posterior"]
            pred_vec = np.array([gp_posterior[name]["mean"] for name in names])
            dist = np.linalg.norm(pred_vec - front_means) / max(1e-6, np.sqrt(len(names)))
            distances_to_mean.append(dist)
        # Normalize and compute entropy-like penalty
        if len(context["pool"]) > 0:
            norm_distances = (np.array(distances_to_mean) - min(distances_to_mean)) / (
                max(1e-9, max(distances_to_mean) - min(distances_to_mean))
            )
            entropies = np.exp(-norm_distances)
        else:
            entropies = [0.0]*len(context["pool"])
    
    # Blend acquisition value with entropy penalty
    scores = []
    progress_factor = context['campaign']['progress']
    for i, cand in enumerate(context["pool"]):
        acq_normed = cand["acq_value_norm"]
        uncertainty_bonus = entropies[i] * (1.0 - 2*abs(progress_factor - 0.5)) # more uncertain early/late
        score = max(0., acq_normed + 0.3 * uncertainty_bonus)
        scores.append(score)

    return scores