def score_pool(context):
    """Estimate hypervolume improvement potential by resampling candidate predictions and scoring based on how much they expand the Pareto front."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    pf = context["pareto_front"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample candidate's objectives multiple times to estimate front expansion
        n_samples = 100
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"] 
            std_val = gp[name]["std"]
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
        
        # For each sample, compute hypervolume contribution
        hv_contributions = []
        for i_sample in range(n_samples):
            candidate_obj = samples[i_sample]
            
            # If dominated by current front or outside reference point, no HV benefit  
            if np.any(candidate_obj >= ref_point) or any(np.all(candidate_obj <= pf_row) and 
                not all(candidate_obj == pf_row) for pf_row in pf):  # dominated
                hv_contributions.append(0.0)
            
            else:
                # Compute hypervolume of the region that would be added by this candidate if it were non-dominated  
                expanded_front = np.vstack([pf, candidate_obj])
                
                # Use a simple method to approximate HV: check how much more area is covered
                try:
                    hv_improvement = 1.0 
                    for j in range(len(names)):
                        max_val_in_pf = pf[:,j].max() if len(pf) > 0 else ref_point[j] - 5 # dummy lower bound  
                        min_ref = candidate_obj[j]
                        
                        width_j = (ref_point[j]) - np.max([min_ref, max_val_in_pf])
                        hv_improvement *= width_j
                    hv_contributions.append(hv_improvement)
                except:
                    hv_contributions.append(0.0)

        # Use mean of HV contributions as the score for this candidate 
        scores.append(np.mean(hv_contributions) if len(hv_contributions)> 0 else 0.)

    return scores