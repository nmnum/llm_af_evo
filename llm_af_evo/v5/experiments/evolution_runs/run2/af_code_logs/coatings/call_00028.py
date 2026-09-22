def score_pool(context):
    """Estimate each candidate’s potential to expand hypervolume and penalize similarity with top candidates."""
    names = context["objective_names"]
    
    # Start with acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Estimate how much each candidate would improve HV if selected
    hv_improvements = []
    front_range = context["pareto_front_range"]
        
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]

        # Compute mean objective values 
        means = [gp_posterior[name]["mean"] for name in names]
            
        hv_improvements.append(means[0])  # rough proxy: use first obj's predicted value

    scores = []
    
    top_candidates_idx = np.argsort(acq_scores)[-5:] if len(context["pool"]) >= 5 else range(len(context["pool"]))
        
    for i, cand in enumerate(context["pool"]):
        base_score = acq_scores[i]
            
        # Add a penalty based on how close this candidate is to already selected top candidates
        repulsion_penalty = 0.0

        if len(top_candidates_idx) > 1:
            current_x = np.array(cand['x'])
                
            for j in top_candidates_idx: 
                if i == j or not (j < len(context["pool"])):
                    continue
                    
                other_cand = context["pool"][j]
                other_x = np.array(other_cand['x'])

                # Compute Euclidean distance between candidates
                dist = np.linalg.norm(current_x - other_x)
                
                # Apply inverse-distance penalty to avoid clustering 
                repulsion_penalty += 1.0 / (dist + 1e-8) 

        final_score = base_score * (1.0 + 0.5*repulsion_penalty)

        scores.append(final_score)
        
    return scores