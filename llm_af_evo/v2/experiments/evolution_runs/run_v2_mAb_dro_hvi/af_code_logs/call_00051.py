def score_pool(context):
    """Estimate expected hypervolume improvement using Monte Carlo samples from each candidate's GP posteriors, plus uncertainty-weighted novelty bonus."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    front_range = context["pareto_front_range"]
    
    # Sample candidates' objectives to estimate HV improvement
    n_samples = 100
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        samples = np.zeros((n_samples, len(names)))
        
        # Draw from each objective's GP posterior
        for i, name in enumerate(names):
            mean, std = gp[name]["mean"], gp[name]["std"]
            samples[:,i] = np.random.normal(mean, std, n_samples)
            
        # Compute hypervolume improvement over current front using these samples 
        hv_improvements = []
        for sample_obj in samples:
            if all(sample_obj >= ref_point):  # dominates reference
                continue
                
            # Use the sampled point to compute HV contribution (simplified estimation)  
            contrib_hv = np.prod(np.maximum(ref_point - sample_obj, 0))
            hv_improvements.append(contrib_hv)
            
        expected_hvi = np.mean(hv_improvements) if len(hv_improvements) > 0 else 0.0

        # Add uncertainty bonus (novelty component based on distance from observed points and stds)
        x_cand = cand["x"]
        dist_to_observed = min(np.linalg.norm(x_cand - obs_x, axis=0).max() for obs_x in context["X_obs"]) if len(context["X_obs"]) > 0 else float('inf')
        
        # Weight uncertainty by how far we are from previously observed points
        novelty_bonus = (1. / (dist_to_observed + 1e-6)) * sum(gp[name]["std"] for name in names) 

        scores.append(expected_hvi + 2.0 * novelty_bonus)
    
    return scores