def score_pool(context):
    """Estimate Pareto optimality probability and penalize batch redundancy to improve exploration-exploitation balance."""
    names = context["objective_names"]
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    
    # Estimate each candidate's chance of being Pareto-optimal using Gaussian sampling
    n_samples = 100
    pareto_probs = []
    front_range = context["pareto_front_range"]
    
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        
        # Sample from the joint posterior distribution of objectives 
        samples = np.zeros((n_samples, len(names)))
        for j, name in enumerate(names):  
            mean = gp_posterior[name]["mean"]
            std = gp_posterior[name]["std"]   
            samples[:,j] = np.random.normal(mean, std, n_samples)
        
        # Count how many sampled points are non-dominated by current front
        pareto_count = 0 
        for sample in samples:
            is_pareto = True
            for point in context["pareto_front"]:
                if all(sample[k] <= point[k] + 1e-8 for k in range(len(names))) and \
                   any(sample[k] < point[k] - 1e-8 for k in range(len(names))):
                    is_pareto = False
                    break 
            if is_pareto:
                pareto_count += 1
        
        prob Pareto_optimal = float(pareto_count) / n_samples  
        # Scale by acquisition score to favor high-acq candidates that are also likely optimal
        pareto_probs.append(prob_Pareto_optimal * acq_scores[i])
    
    # Add a diversity penalty based on proximity of top-ranked candidates 
    batch_size = min(5, len(context["pool"]))
    if len(acq_scores) >= 2:
        
        sorted_indices = np.argsort(-acq_scores)[:batch_size]
  
        repulsion_penalty = []
        for i in range(len(context["pool"])):
            penalty = 0.0
            cand_x = context["pool"][i]["x"]
            
            # Compute average distance to top candidates (excluding self)
            distances_to_top = [] 
            for j_idx, idx_j in enumerate(sorted_indices):
                if i != idx_j:
                    dist_sq = np.sum((cand_x - context["pool"][idx_j]["x"])**2)  
                    distances_to_top.append(dist_sq)

            # If candidate is close to top-ranked ones (i.e. low diversity), penalize
            avg_dist = 0.
            if len(distances_to_top):
                avg_dist = np.mean(distances_to_top)
            
            repulsion_penalty.append(-avg_dist * 0.1)  
        final_scores = acq_scores + pareto_probs + np.array(repulsion_penalty)

    else:
        
        # Fallback to basic acquisition score only
        final_scores = list(acq_scores)

    return [float(s) for s in final_scores]