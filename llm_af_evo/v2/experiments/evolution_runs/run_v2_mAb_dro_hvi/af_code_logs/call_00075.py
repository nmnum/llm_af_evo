def score_pool(context):
    """Estimate improvement potential using hypervolume contribution sampled from GP posteriors."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample objective values from the candidate's GP posterior
        n_samples = 100
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"] 
            std_val = gp[name]["std"]
            samples[:,i] = np.random.normal(mean_val, std_val, size=n_samples)
        
        # Compute hypervolume contribution of each sample
        hv_contribs = []
        for smp in samples:
            if all(smp >= ref_point):  # dominated by reference point 
                continue
            
            # Calculate HV improvement using current pareto front and this candidate's prediction  
            expanded_front = np.vstack([context["pareto_front"], smp])
            
            # Simple hypervolume calculation (assuming two objectives for simplicity)
            if len(names) == 2:
                hv_improvement = (
                    max(ref_point[0], expanded_front[:,0].max()) - ref_point[0]) * \
                    (max(ref_point[1], expanded_front[:,1].max()) - ref_point[1])
                
                # Subtract the existing hypervolume to get net gain
                current_hv = np.prod(context["pareto_front_range"][name] for name in names)  if len(context["pareto_front"]) > 0 else 0
                
            hv_contribs.append(hv_improvement)
        
        score = float(np.mean(hv_contribs)) if hv_contribs else -1e6
        scores.append(score)

    return scores