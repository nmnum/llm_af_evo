def score_pool(context):
    """Estimate improvement potential by resampling predicted objectives around each candidate's mean, then compute hypervolume contribution of those samples."""
    names = context["objective_names"]
    front = context["pareto_front"] 
    ref_point = context["ref_point"]

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample from each objective's GP posterior (mean, std)
        n_samples = 10
        samples_per_obj = [np.random.normal(gp[name]["mean"], gp[name]["std"], size=n_samples) 
                           for name in names]
                
        # For each sample point across all objectives:
        hv_contributions = []
        for i_sample in range(n_samples):
            sampled_point = np.array([samples[i_sample] for samples in samples_per_obj])
            
            # Compute hypervolume contribution of this single point
            if not any(sampled_point < front[:, j].min() or 
                       sampled_point[j] > ref_point[j]
                       for j in range(len(names))):
                hv_contrib = 1.0  
                for k, val in enumerate(sampled_point):
                    hv_contrib *= (ref_point[k] - max(val, front[:,k].max()))
                
                # Only count if point is not dominated
                dominates_any_front = any(
                    all(front[i][j] >= sampled_point[j] 
                        for j in range(len(names))) and  
                    any(front[i][j] > sampled_point[j]
                         for j in range(len(names)))
                     for i in range(len(front))
                 )
                
                if not dominates_any_front:
                    hv_contributions.append(hv_contrib)
        
        # Score is average hypervolume contribution across all samples
        score = np.mean(hv_contributions) if len(hv_contributions)>0 else 0.0
        
        scores.append(score)

    return scores