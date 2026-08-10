def score_pool(context):
    """Exploitation-aware uncertainty scoring with dynamic reference point adjustment based on observed hypervolume expansion potential."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    
    # Estimate how much each candidate would improve the current hypervolume
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted means and stds normalized to front range
        mu_normalized = [gp[name]["mean"] / front_range[name] for name in names]
        sigma_normalized = [gp[name]["std"] / front_range[name] for name in names]

        # Adjust reference point dynamically based on current pareto_front spread (more aggressive exploration early)
        adjusted_ref_point = ref_point.copy()
        
        if context["campaign"]["progress"] < 0.3:
            # Early stage: more exploratory, push towards under-explored regions
            for i in range(len(names)):
                if mu_normalized[i] > np.mean([pf[i]/front_range[name] for pf in context['pareto_front']]):
                    adjusted_ref_point[i] = ref_point[i]
        else:
            # Later stage: exploit known good areas, less uncertainty tolerance 
            pass  # Keep original reference point

        # Hypervolume improvement estimate using Monte Carlo sampling of the candidate's GP posterior
        n_samples = min(1000, max(len(context["Y_obs"]), 25))
        
        hv_improvement_estimate = []
        for _ in range(n_samples):
            sample_mu = [np.random.normal(mu_normalized[i], sigma_normalized[i]) 
                         if sigma_normalized[i] > 0 else mu_normalized[i]
                         for i in range(len(names))]
            
            # Ensure the sampled point is above reference
            hv_improvement_estimate.append(
                max(1e-8, np.prod([max(sample_mu[j]-adjusted_ref_point[j], 0) 
                                   if adjusted_ref_point[j] > sample_mu[j] else (sample_mu[j])  
                                  for j in range(len(names))]))
            )

        # Use mean of HV improvement estimates as score
        scores.append(np.mean(hv_improvement_estimate))

    return scores