def score_pool(context):
    """
    Estimate hypervolume improvement by resampling candidates' objectives from their GPs,
    then rank based on how much each candidate would expand the dominated region if its 
    prediction were correct, using a noisy Pareto front to avoid overfitting.
    """
    names = context["objective_names"]
    ref_point = np.array([context['ref_point_by_name'][name] for name in names])
    
    # Resample from GP posteriors and compute empirical hypervolume
    n_samples = 50
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        samples = np.array([[np.random.normal(gp[name]["mean"], gp[name]["std"]) 
                             for name in names] 
                            for _ in range(n_samples)])
        
        # Use current Pareto front + noise to estimate expansion
        pf_with_noise = context['pareto_front'] + np.random.randn(*context['pareto_front'].shape) * 0.1
        
        hv_improvement = []
        cand_pred = [gp[name]["mean"] for name in names]
        
        # For each sample, compute hypervolume if this candidate was added
        for smp in samples:
            combined_pf = np.vstack([pf_with_noise, smp])
            
            # Find non-dominated points (this is a simplified version)
            nondom_mask = ~np.any(np.all(combined_pf[None,:] >= combined_pf[:, None], axis=2), axis=0) 
            pf_new = combined_pf[nondom_mask]
            
            if len(pf_new) > 1:
                hv_improvement.append(
                    np.prod(ref_point - np.min(pf_new, axis=0))
                )
        
        # Average HV improvement across samples
        avg_hv_imp = sum(hv_improvement)/len(hv_improvement) if hv_improvement else 0.0
        
        scores.append(avg_hv_imp)
    
    return scores