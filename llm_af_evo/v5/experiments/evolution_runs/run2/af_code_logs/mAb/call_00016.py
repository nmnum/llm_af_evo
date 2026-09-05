def score_pool(context):
    """Suppress scores of candidates that are too close to already selected high-scoring ones, encouraging batch-wide diversity."""
    names = context["objective_names"]
    
    # Compute GP means for all candidates 
    gp_means = np.array([[cand["gp_posterior"][name]["mean"] for name in names] for cand in context["pool"]])
    
    scores = []
    for i, cand in enumerate(context["pool"]):
        acq_value_norm = cand["acq_value_norm"]
        
        # Start with acquisition value
        score = acq_value_norm
        
        # If this candidate is near any previously selected top candidates (within 10% of front range),
        # suppress its score significantly to encourage diversity in the batch.
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Check if there's a high-scoring candidate nearby
        threshold_dist = min(context["pareto_front_range"].values()) * 0.1
        
        for j, prev_gp_mean in enumerate(gp_means[:i]):   # Only check previously scored candidates  
            dist_to_prev = np.linalg.norm(prev_gp_mean - gp_mean)
            
            if dist_to_prev < threshold_dist:
                score *= 0.3    # Suppress the score significantly to avoid duplicates
                break
                
        scores.append(score)

    return scores