def score_pool(context):
    """Estimate hypervolume expansion potential using noisy posterior sampling and penalize candidates near existing observations."""
    n_samples = 10
    ref_point = context["ref_point"]
    X_obs = context["X_obs"] 
    names = context["objective_names"]

    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample noisy objectives from GP posterior  
        samples = np.array([
            [np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"]) 
             for name in names]
            for _ in range(n_samples)
        ])
        
        # Compute hypervolume improvement estimates
        volumes = []
        for s in samples:
            vol = max(0, np.prod(s - ref_point))
            volumes.append(vol)

        mean_vol = np.mean(volumes) 
        std_vol = np.std(volumes)
            
        # Add a novelty penalty based on distance to nearest observation  
        cand_x = cand["x"]
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            min_dist = np.min(distances)
            novel_penalty = max(0.0, 1.0 / (min_dist + 1e-8)) 
        else:  
            novel_penalty = 1.0

        # Final score is mean volume minus uncertainty scaled by novelty
        scores.append(mean_vol - std_vol * novel_penalty)

    return scores