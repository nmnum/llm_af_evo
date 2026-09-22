def modifier(context):
    """Add uncertainty bonus scaled by how close candidates are to existing observations, promoting exploration of under-represented regions."""
    
    import numpy as np
    
    names = context["objective_names"]
    if len(context.get("X_obs", [])) == 0:
        return [0.] * len(context["pool"])
        
    values = []
    n_samples = 64
    weight_factor = 1.5 * (1 - context["campaign"]["progress"]) 
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        means = np.array([gp_posterior[name]["mean"] for name in names])
        stds = np.array([gp_posterior[name]["std"] for name in names])

        try:
            cov_matrix = np.diag(stds**2)
            sampled_means = np.random.multivariate_normal(means, cov_matrix, n_samples) 
        except:  
            sampled_means = np.tile means, (n_samples,1)

        # Compute distance to nearest observation
        cand_x = cand["x"]
        distances = [np.linalg.norm(cand_x - obs_x) for obs_x in context["X_obs"]]
        min_distance = np.min(distances)
        
        uncertainty_bonus = weight_factor * stds.mean()
        novelty_penalty = 0.5 * (1 / (min_distance + 1e-6))
    
        values.append(uncertainty_bonus - novelty_penalty)

    return values