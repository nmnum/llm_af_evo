def score_pool(context):
    """Exploitation with uncertainty-aware hypervolume improvement estimation: candidates are scored by how much they would expand the dominated region if their predictions were correct."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        # Predicted means and stds
        mu = [gp[name]["mean"] for name in names]
        sigma = np.array([gp[name]["std"] for name in names])

        # Estimate hypervolume improvement by sampling from the candidate's posterior (MC)
        n_samples = 100 
        samples = []
        for _ in range(n_samples):
            sample_mu = mu + np.random.randn(len(names)) * sigma
            samples.append(sample_mu)

        hv_improvement_estimate = 0.0
        
        # Compute hypervolume improvement relative to current pareto front using the sampled points  
        if len(context["pareto_front"]) > 1:
            for sample in samples: 
                dominated_by_current_pf = False
                for pf_point in context["pareto_front"]:
                    if all(pf_point >= sample):  # Sample is dominated by PF point (assuming maximization)
                        dominated_by_current_pf = True  
                        break

                # If the sampled candidate extends beyond current front, it adds to HV 
                hv_improvement_estimate += float(not dominated_by_current_pf)

        scores.append(hv_improvement_estimate / n_samples)  # Normalize by number of samples
    
    return scores