def score_pool(context):
    """Estimates pareto optimality by resampling each candidate's GP posteriors, then scores based on how frequently sampled points dominate the current front."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Resample from each candidate's posterior to estimate dominance
    n_samples = 50
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample objectives for this candidate
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"] 
            std_val = gp[name]["std"]
            if std_val > 0:
                samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
            else:
                samples[:,i] = mean_val
        
        # Count how many sampled points dominate the current front
        dominance_count = sum(
            not any(np.all(samples[j,:] <= pf_point) for pf_point in context["pareto_front"])
            for j in range(n_samples)
        )
        
        scores.append(dominance_count / n_samples)

    return scores