def score_pool(context):
    """Estimates hypervolume contribution using noisy GP samples and rewards diverse, high-impact predictions."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    # Sample from posteriors to estimate HV improvement
    n_samples = 100
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Draw samples for each objective
        samples = np.array([
            np.random.normal(gp[name]["mean"], gp[name]["std"], size=n_samples)
            for name in names]).T

        # Compute hypervolume contribution of this candidate's sampled points
        hv_contributions = []
        
        for sample in samples:
            if all(sample >= ref_point):  # Dominated by reference point?
                continue
            
            # Calculate HV improvement: volume between current front and extended with new point
            expanded_ref = np.minimum(ref_point, sample)
            
            # Simple estimation of hypervolume difference (approximation via min distances to pareto_front in each dimension) 
            if len(context["pareto_front"]) > 0:
                pf_min_dists = [min(abs(sample[i] - pfd) for pfd in context["pareto_front"][:, i]) 
                                for i in range(len(names))]
                
                # Use product of distances to approximate volume increase
                hv_contributions.append(np.prod(pf_min_dists))
            else:
                # No front yet, so just use the inverse distance from reference point (simplified)
                dist_to_ref = np.linalg.norm(sample - ref_point) 
                if not np.isclose(dist_to_ref, 0):
                    hv_contributions.append(1. / dist_to_ref)

        score = float(np.mean(hv_contributions)) if len(hv_contributions) > 0 else 0.
        
        # Add a diversity bonus to avoid selecting very similar candidates
        x_cand = cand["x"]
        min_dist = np.linalg.norm(context["X_obs"] - x_cand, axis=1).min() + 1e-8
        
        score += max(0., (5. / (min_dist * len(context["pool"])))) # bonus inversely proportional to proximity

        scores.append(score)

    return scores