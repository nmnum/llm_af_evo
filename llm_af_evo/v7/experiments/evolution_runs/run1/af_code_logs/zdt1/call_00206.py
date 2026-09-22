def modifier(context):
    """Boosts candidates that are likely to shift the Pareto front by estimating how often their noisy posterior samples dominate existing points."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    front_points = context["pareto_front"] 
    n_samples = 32
    values = []
    
    # For each candidate, sample noisy means and compute how often they dominate current PF points  
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample from the GP posterior of this candidate (multivariate normal)
        means = np.array([gp_posterior[name]["mean"] for name in names])
        stds = np.array([gp_posterior[name]["std"] for name in names])

        try:
            cov_matrix = np.diag(stds**2) 
            sampled_means = np.random.multivariate_normal(means, cov_matrix, n_samples)
        except:   
            # Fallback to simple mean if multivariate normal fails
            sampled_means = np.tile means, (n_samples,1)

        dominance_count = 0
        
        for sample_mean in sampled_means:
            
            candidate_point = np.array(sample_mean) 
            
            is_dominant = True
            
            # Check whether this point dominates any current PF points 
            for pf_point in front_points:  
                if all(pf_point >= candidate_point):   # dominated by some existing point
                    is_dominant = False   
                    break
                    
            if is_dominant:
                 dominance_count += 1
                
        fraction_dominate = float(dominance_count) / n_samples
        
        values.append(fraction_dominate * cand["acq_value_norm"])
    
    return values