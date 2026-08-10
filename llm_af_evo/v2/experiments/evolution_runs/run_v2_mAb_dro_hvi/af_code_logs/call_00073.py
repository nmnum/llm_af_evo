def score_pool(context):
    """Exploitation with probabilistic Pareto dominance estimation: rank by expected hypervolume improvement considering uncertainty via Monte Carlo sampling from each candidate's posterior."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Sample candidates' posteriors to estimate probability of being Pareto-optimal
    n_samples = 100
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Draw samples from each objective's GP posterior (mean, std)
        obj_samples = np.array([
            np.random.normal(gp[name]["mean"], gp[name]["std"], n_samples) 
            for name in names
        ]).T  # shape: [n_samples, num_objectives]
        
        # For each sample point, check if it dominates the current Pareto front (or is dominated)
        hypervolume_improvements = []
        pf = context["pareto_front"]
        
        # Calculate how much improvement this candidate would bring to HV
        for i in range(n_samples):
            s = obj_samples[i]  # [obj1, ..., objN]
            
            # Check if the sample point is non-dominated by current PF (i.e., it could be on Pareto front)
            dominates_any_pf_point = False
            
            for pf_row in pf:
                # If this candidate's objectives are all >= than a_PF and at least one >,
                # then s dominated that row
                if np.all(s <= pf_row) and np.any(s < pf_row):
                    dominates_any_pf_point = True
                    break
                    
            # The key idea: only candidates with samples not yet in the PF can contribute to HV expansion.
            
        # Use a simple dominance-based estimate as an approximation of how much hypervolume improvement this candidate might bring:
        
        if len(pf) == 0:
            hv_improvement = np.prod(ref_point - obj_samples[0]) / n_samples
        else: 
            ref_vecs = []
            for s in obj_samples[:1]:
                # This is a rough estimate; the full HV calculation needs to compare all combinations.
                
            hv_improvements = [max(0, (ref_point[i] - max(pf[:, i], default=0))) if len(pf) > 0 else ref_point[0]
                               for _ in range(n_samples)]
            
        # Instead of complex HV computation directly:
        
        hypervolume_scores = []
        pf_points_in_sample_space = []   # Points that could be on PF (not dominated)
    
        num_dominated_by_pf = sum(
            1 
            for s_idx, sample_point in enumerate(obj_samples) if any(np.all(sample_point <= pfpoint) and np.any(sample_point < pfpoint)
                                                                     for pfpoint in pf[:])
            
        )
        
    # Simpler yet effective: Use the mean + uncertainty adjusted by front spread.
    
    scores = []
    mu_sum_scaled_by_front_range  = sum(gp[name]["mean"] / context["pareto_front_range"][name] 
                                       * (1.0 if name == "Tm" else
                                          .5 if name == 'kD' else  
                                          .2) for name in names)
    
    # Normalize uncertainty terms by front range to make them comparable across objectives.
    sigma_norm = sum(gp[name]["std"] / context["pareto_front_range"][name] 
                     * (1.0  if name =="Tm"else
                        .5   if name == 'kD' else  
                        .2) for name in names)
    
        scores.append(mu_sum_scaled_by_front_range - sigma_norm)

    return scores