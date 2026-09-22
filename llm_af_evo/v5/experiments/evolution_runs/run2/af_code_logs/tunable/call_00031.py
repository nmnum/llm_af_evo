def score_pool(context):
    """Estimate pareto-optimality and use repulsive scoring to encourage diverse exploration."""
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Base acquisition values (already hypervolume improvement estimates)
    base_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Estimate probability of being Pareto-optimal using noise-aware GP samples
    pareto_probs = []
    n_samples = 10
    
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        
        # Sample from the joint posterior (simplified: sample each objective independently)
        sampled_values = np.zeros((n_samples, len(names)))
        for j, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"] 
            sampled_values[:,j] = np.random.normal(mean_val, std_val, n_samples)

        # Count how often this candidate dominates others (in the sample)
        dominated_count = 0
        for j in range(n_samples):
            is_dominated_by_any = False
            
            # Compare against all other candidates' samples 
            for k, cand_k in enumerate(context["pool"]):  
                if i == k:
                    continue
                    
                gp_posterior_k = cand_k["gp_posterior"]
                
                sampled_values_k = np.zeros((n_samples, len(names)))
                for l, name_l in enumerate(names):
                    mean_val_k = gp_posterior_k[name_l]["mean"]
                    std_val_k = gp_posterior_k[name_l]["std"] 
                    sampled_values_k[:,l] = np.random.normal(mean_val_k, std_val_k, n_samples)
                
                # Check if current sample is dominated by this candidate
                for s in range(n_samples):
                    dom_by_samp = True  
                    
                    for l, name_l in enumerate(names):  # all objectives better or equal?
                        if sampled_values[s,l] < sampled_values_k[s,l]: 
                            dom_by_samp = False
                            break
                        
                    if not (dom_by_samp and any(sampled_values[s,:] > sampled_values_k[s,:])):
                        continue
                    
                is_dominated_by_any = True  
                
            dominated_count += int(is_dominated_by_any)
            
        pareto_probs.append(1.0 - float(dominated_count) / n_samples)

    # Compute novelty as minimum distance to already observed points
    x_observed = context["X_obs"]
    
    if len(x_observed):
        novelties = []
        
        for cand in context["pool"]:
            dists_to_observed = np.linalg.norm(cand['x'] - x_observed, axis=1)
            min_dist = np.min(dists_to_observed) 
            # Invert so higher novelty means farther from existing points
            novelties.append( 1.0 / (min_dist + 1e-8))
    else:
        novelties = [1.] * pool_size

    final_scores = []
    
    for i in range(pool_size):
        
        base_score = base_scores[i]
        pareto_prob = pareto_probs[i] 
        novelty = novelties[i]

        # Combine acquisition, probability of being Pareto-optimal and novelty
        score = (base_score * 0.7 +  
                 pareto_prob * 0.2 +
                novelty   * 0.1)
        
        final_scores.append(score)

    return final_scores