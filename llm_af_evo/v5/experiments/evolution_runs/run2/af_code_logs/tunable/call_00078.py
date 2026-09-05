def score_pool(context):
    """Blend acquisition value with an uncertainty-adjusted front proximity signal that rewards candidates near sparsely-covered frontier regions and penalizes those in high-uncertainty areas of dense coverage."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    ref_point = context["ref_point"]
    
    # Compute normalized hypervolume contributions for each candidate
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Acquisition value (already computed)
        acq_value_norm = cand['acq_value_norm']

        # Estimate how far the predicted objective is from current Pareto front,
        # weighted by uncertainty: lower uncertainty near a frontier region should
        # increase score more than high-uncertainty points.
        mean_vector = np.array([gp[name]["mean"] for name in names])
        
        if len(pf) >= 1:
            # Compute distances to the nearest point on Pareto front (for each candidate)
            min_dist_to_front = float('inf')
            
            for pf_point in pf:
                dist_sq = sum((a - b)**2 for a, b in zip(mean_vector, pf_point))
                if dist_sq < min_dist_to_front:
                    min_dist_to_front = dist_sq
            
            # Normalize distance to front by the range of each objective
            normalized_distance = np.sqrt(min_dist_to_front) / (
                sum((context["pareto_front_range"][name])**2 for name in names)**0.5 + 1e-8)
        else:
            normalized_distance = float('inf')

        # Compute uncertainty penalty: higher std -> lower score boost
        sigma_sum_normed = sum(gp[name]["std"] / context["pareto_front_range"][name] 
                               for name in names)

        # Final combined signal (uncertainty penalizes proximity to front)
        base_score = acq_value_norm + 1.0 * np.exp(-normalized_distance) - \
                     sigma_sum_normed

        scores.append(base_score)

    return scores