def score_pool(context):
    """Estimate hypervolume improvement potential using Monte Carlo samples from each candidate's GP posteriors."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    front_range = context["pareto_front_range"]
    
    # Sample objectives for each candidate
    n_samples = 100
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Draw samples from the joint posterior of all three objectives
        means = [gp[name]["mean"] for name in names]
        stds = [gp[name]["std"] for name in names]
        cov_matrix = np.diag(stds) @ np.eye(3)  # Simplified: assuming independence
        
        try:
            samples = np.random.multivariate_normal(means, cov_matrix, n_samples)
            
            # Compute hypervolume improvement estimate
            hv_improvements = []
            for sample in samples:
                if all(sample >= ref_point): continue
                
                temp_front = context["pareto_front"].copy()
                
                # Add the sampled point to front and compute new HV (simplified estimation)  
                extended_front = np.vstack([temp_front, sample])
                dominated_mask = ~np.any(np.all(extended_front[None,:] <= extended_front[:, None], axis=2), axis=1)
                pareto_points = extended_front[dominated_mask]
                
                if len(pareto_points) > 0:
                    hv_improvement = np.prod(ref_point - np.min(pareto_points, axis=0))
                    
                    # Normalize by the front range
                    norm_factor = np.product(front_range.values())
                    normalized_hv_imp = hv_improvement / (norm_factor + 1e-8)
                else:
                    normalized_hv_imp = 0.0
                    
                hv_improvements.append(normalized_hv_imp)

            score = float(np.mean(hv_improvements)) if len(hv_improvements) > 0 else -np.inf
        except Exception as e: 
            # Fallback to simple mean for robustness  
            mu_sum = sum(gp[name]["mean"] for name in names)
            sigma_norm = np.sum([gp[name]["std"]/front_range[name] for name in names])
            
            score = 0.9 * (mu_sum + 1e-8) - 2.5*sigma_norm

        scores.append(score)

    return scores