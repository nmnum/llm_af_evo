def score_pool(context):
    """Estimate improvement potential using hypervolume expansion adjusted for uncertainty; prioritize candidates that could dominate or significantly expand the Pareto front."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted means and stds
        mu = [gp[name]["mean"] for name in names]
        sigma = [gp[name]["std"] for name in names]

        # Estimate hypervolume improvement using Monte Carlo sampling of the posterior.
        n_samples = 100
        
        hv_improvements = []
        
        for _ in range(n_samples):
            sample_mu = np.array(mu) + np.random.randn(len(names)) * sigma
            if all(sample_mu[i] <= ref_point[i] for i in range(len(names))):
                # The sampled point is dominated by the reference, so it contributes 0 HV improvement.
                hv_improvements.append(0.0)
            else:
                sample_ref = np.minimum(ref_point, sample_mu) 
                hypervolume_gain = max(np.prod(sample_ref - ref_point), 0.)
                hv_improvements.append(hypervolume_gain)

        # Average the estimated HV improvements over all samples.
        score = sum(hv_improvements)/n_samples

        scores.append(score)
        
    return scores