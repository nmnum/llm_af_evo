def score_pool(context):
    """Estimate hypervolume improvement potential using Monte Carlo samples from each candidate's GP posteriors, then normalize by current front spread."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    pareto_front = context["pareto_front"]
    front_range = np.array([context["pareto_front_range"][name] for name in names])

    # Sample from each candidate's GP posterior to estimate hypervolume improvement
    n_samples = 100
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Draw samples from the joint distribution of all objectives for this candidate
        means = np.array([gp[name]["mean"] for name in names])
        stds = np.array([gp[name]["std"] for name in names])

        # Use Cholesky decomposition to sample correlated outcomes (approximate)
        try:
            cov_matrix = np.diag(stds ** 2) 
            L = np.linalg.cholesky(cov_matrix)

            samples = means + np.random.normal(size=(n_samples, len(names))) @ L.T
        except np.linalg.LinAlgError: # If Cholesky fails due to numerical issues:
            samples = np.tile(means, (n_samples, 1))

        # Compute hypervolume improvement for each sample 
        hv_improvements = []
        
        for s in samples:
            
            proposed_point = s
            
            if len(pareto_front) == 0:  
                front_hv_contrib = max(0., np.prod(ref_point - proposed_point))
                
            else:

                dominated_by_current_pf = False
                # Check domination against current Pareto front (domination in the opposite sense)
                for pf_pt in pareto_front:
                    if all(proposed_point[i] >= pf_pt[i] for i in range(len(names))):
                        dominates_this_candidate = True  
                        
                        break
                        
            hv_improvements.append(max(0., np.prod(ref_point - proposed_point)))
        
        # Take the mean HV improvement across samples
        scores.append(np.mean(hv_improvements))
    
    return scores