def score_pool(context):
    """Estimates hypervolume improvement potential using noisy GP sampling and penalizes candidates near existing observations."""
    names = context["objective_names"]
    ref_point = context["ref_point"]
    X_obs = context["X_obs"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Estimate HV improvement by resampling the GP posteriors under noise
        hv_improvement_estimates = []

        # Sample 10 times from each objective's posterior to estimate potential HV gain
        n_samples = 10

        for _ in range(n_samples):
            sampled_objectives = []
            for name in names:
                mean, std = gp[name]["mean"], gp[name]["std"]
                sample = np.random.normal(mean, std)
                sampled_objectives.append(sample)

            # Compute hypervolume contribution of this sample
            if all(sampled_obj >= ref_point[i] for i, sampled_obj in enumerate(sampled_objectives)):
                hv_contribution = 1.0
                for i, (sampled_obj, rp) in enumerate(zip(sampled_objectives, ref_point)):
                    # Simplified hypervolume calculation: product of differences from reference point.
                    hv_contribution *= max(0., sampled_obj - rp)
            else:
                hv_contribution = 0.0

            hv_improvement_estimates.append(hv_contribution)

        estimated_hypervolume_gain = np.mean(hv_improvement_estimates) if len(hv_improvement_estimates) > 0 else 0.

        # Penalize candidates that are too close to already observed points
        cand_x = cand["x"]
        min_distance_to_observed = float('inf')
        
        for obs in X_obs:
            dist_sq = np.sum((cand_x - obs)**2)
            if dist_sq < min_distance_to_observed:
                min_distance_to_observed = dist_sq

        # Inverse distance penalty: larger penalties when close to existing points
        novelty_penalty_factor = 1.0 / (min_distance_to_observed + 1e-8)

        score = estimated_hypervolume_gain * novelty_penalty_factor
        
        scores.append(score)
        
    return scores