def score_pool(context):
    """Estimate probability of acandidatebeingParetooptimalandblendwithacquisitionvalueforrobustexploration-exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute probability of being Pareto optimal using Gaussian approximation
    pareto_probs = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Get mean and std for each objective  
        means = np.array([gp_posterior[name]["mean"] for name in names])
        sigmas = np.array([gp_posterior[name]["std"] for name in names])

        # Simple heuristic: if all objectives are above reference point, 
        # candidate is likely not dominated; otherwise compute a rough dominance probability
        dominates_ref = np.all(means > ref_point)
        
        prob_pareto = 1.0 if dominates_ref else 0.5
        
        # Adjust based on how close we are to the current Pareto front  
        for obj_name in names:
            mean_obj, std_obj = gp_posterior[obj_name]["mean"], gp_posterior[obj_name]["std"]
            
            ref_val = context["ref_point_by_name"][obj_name]
                
            if (mean_obj > ref_val):  # If above reference point
                prob_pareto *= max(0.1, 
                    np.exp(-((mean_obj - ref_val) / std_obj)**2))  
        
        pareto_probs.append(prob_pareto)

    acq_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Blend acquisition value with Pareto probability
    scores = []
    alpha, beta= 0.75, 1 - 0.75 
    for i in range(len(context["pool"])):  
        score = (alpha * acq_scores[i] + beta * pareto_probs[i])
        scores.append(score)
        
    return scores