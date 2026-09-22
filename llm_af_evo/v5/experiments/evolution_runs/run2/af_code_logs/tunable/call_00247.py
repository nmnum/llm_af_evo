def score_pool(context):
    """Estimate each candidate’s potential for discovering novel non-dominated points by measuring how much their uncertainty overlaps with uncovered regions of objective space."""
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    ref_point = np.array(context["ref_point"])
    pareto_front = context["pareto_front"]

    # Compute hypervolume contributions for each candidate
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Get predicted means and stds 
        pred_means = np.array([gp_posterior[name]["mean"] for name in names])
        pred_stds = np.array([gp_posterior[name]["std"] for name in names])

        # Estimate how much this candidate's uncertainty overlaps with current pareto front
        overlap_score = 0.0
        
        if len(pareto_front) > 1:
            cand_lower_bound = pred_means - 2 * pred_stds 
            cand_upper_bound = pred_means + 2 * pred_stds
            
            # Check how much of the candidate's uncertainty region intersects with
            # regions dominated by current pareto front points (i.e., where it could improve)
            
            for pf_point in pareto_front:
                if np.all(pf_point <= ref_point): 
                    is_dominated = True
                    
                    # If this point dominates, check overlap of the candidate's uncertainty region  
                    
                    # Candidate must be better than some PF objective to matter
                    dominated_by_cand = False
                    for i, obj in enumerate(names):
                        if pred_means[i] > pf_point[i]:
                            dominated_by_cand = True 
                            
                    if not dominated_by_cand:
                        continue
                    
            overlap_score += 1.0

        # Reward candidates with high acquisition value and low uncertainty overlapping known front regions
        acq_value_norm = cand["acq_value_norm"]
        
        unc_penalty = np.sum(pred_stds / (front_range[name] + 1e-8) for name in names)
                
        score = acq_value_norm - overlap_score * unc_penalty
        
        scores.append(score)

    return scores