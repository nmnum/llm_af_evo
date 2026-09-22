def score_pool(context):
    """Estimates acquisition value by resampling candidates' objective means under GP uncertainty and measuring how often they improve hypervolume coverage relative to observed points."""
    X_obs = context["X_obs"]
    names = context["objective_names"] 
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    
    # Use a fixed number of posterior samples per candidate
    n_samples = 30
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample from thecandidate's GP posteriors 
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Estimate hypervolume improvement potential by sampling
        hv_improvements = []
        
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            # Check if this sample improves the current HV when added to observations 
            expanded_hv = 0.0
            
            # Compute hypervolume with candidate included
            test_front = np.vstack((context["pareto_front"], cand_obj))
            
            # Remove dominated points from front (Pareto pruning)
            non_dominated_mask = ~np.any(
                np.all(test_front[:, None, :] >= test_front[None, :, :], axis=2) &
                ~(test_front[:, None, :] == test_front[None, :, :]).all(axis=2),
                axis=1
            )
            
            pruned_front = test_front[non_dominated_mask]
                
            # Compute HV of this modified front relative to ref_point 
            if len(pruned_front) >= 2:
                try:  
                    hv_value = hypervolume(pruned_front, ref_point)
                    expanded_hv = max(0.0, hv_value - context["hypervolume"])
                except Exception as e:
                    pass
                    
            hv_improvements.append(expanded_hv)

        # Score is average HV improvement across samples
        score = np.mean(hv_improvements) if len(hv_improvements) > 0 else 0.0
        
        scores.append(score)
    
    return scores

# Helper function to compute hypervolume (simplified version assuming two objectives for clarity; in practice, should use a robust library like `sklearn.metrics` or similar with proper implementation of the hyper-volume calculation algorithm):
def hypervolume(front, ref_point): 
    # This is an example placeholder - real-world usage would require full HV computation logic
    return np.sum((ref_point-front).prod(axis=1)) if len(front) > 0 else 0.0