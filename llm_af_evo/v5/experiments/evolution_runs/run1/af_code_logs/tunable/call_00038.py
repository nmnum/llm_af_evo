def score_pool(context):
    """Score candidates by their potential to create novel non-dominated fronts when combined with observed points."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        # Estimate the candidate's contribution to hypervolume by comparing
        # its predicted objectives against current front and reference point  
        pred_obj = np.array([gp[name]["mean"] for name in names])
        
        # Compute how much better this candidate is than all points on pareto_front,
        # using a distance metric that accounts for the direction of improvement.
        distances_to_front = []
        if len(context["pareto_front"]) > 0:
            front_obj = context["pareto_front"]
            
            # For each point in Pareto front, compute how much better this candidate is
            for pf_point in front_obj: 
                diff_vec = pred_obj - pf_point
                
                # Only consider improvements along objectives that are not worse than reference  
                if all(diff >= 0 or (diff < 0 and abs(diff) <= ref_point[i] * 1e-3)
                       for i, diff in enumerate(diff_vec)):
                    distances_to_front.append(np.linalg.norm(diff_vec))
                    
        # Novelty score based on distance to nearest observed point
        x = cand["x"]
        if len(context["X_obs"]) > 0:
            obs_distances = np.min([np.linalg.norm(x - xi) for xi in context["X_obs"]])
            novelty_score = max(1e-6, (1.0 / (obs_distances + 1e-6)) ** 2)
        else: 
            novelty_score = 1.0
            
        # Use the acquisition value as a baseline
        base_acq = cand["acq_value_norm"]
        
        if len(distances_to_front) > 0:
            min_dist = np.min(distances_to_front)
            
            # Prefer candidates that are far from front (i.e., not dominated by current non-dominated points),
            # and also consider how much better they might be
            score = base_acq * max(1.0, novelty_score) + 2.5 / min_dist if distances_to_front else base_acq  
        elif len(context["pareto_front"]) > 0:
            dist_from_ref_point = np.linalg.norm(pred_obj - ref_point)
            
            # If candidate doesn't dominate anything in current front,
            # score based on how much better it is than reference point (i.e., new frontier expansion potential) 
            score = base_acq * max(1.0, novelty_score + 2e-3 / dist_from_ref_point if dist_from_ref_point > 1e-6 else 5)
        else:
             # No front yet - just use acquisition value and bonus for being novel
            score = (base_acq + np.log(novelty_score)) * max(0.8, novelty_score) 
            
        scores.append(score)

    return scores