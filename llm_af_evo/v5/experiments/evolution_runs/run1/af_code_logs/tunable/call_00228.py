def score_pool(context):
    """Blend acquisition value with uncertainty-aware progress sensitivity and adaptive novelty penalty."""
    names = context["objective_names"]
    
    # Use the provided acq_value_norm directly as quality signal
    qualities = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Normalize qualities to [0, 1] 
    q_min, q_max = qualities.min(), qualities.max()
    if abs(q_max - q_min) < 1e-9:
        norm_qualities = np.full_like(qualities, 0.5)
    else:
        norm_qualities = (qualities - q_min) / (q_max - q_min)

    # Compute uncertainty-aware progress sensitivity
    campaign_progress = context["campaign"]["progress"]
    
    # Early: favor exploration; later: exploit more 
    w_explore = max(0., 1.5 * (1. - campaign_progress))
    w_utility = min(1., 2.* campaign_progress)
    
    ucb_weights = np.array([w_explore + w_utility for _ in range(len(context["pool"]))])
   
    # Compute normalized uncertainty as inverse of std sum
    uncertainties = []
    front_range = context['pareto_front_range']
    for cand in context["pool"]:
        sigma_sum_norm = sum(cand["gp_posterior"][name]["std"] / front_range[name] 
                            for name in names)
        uncertainties.append(sigma_sum_norm)

    ucb_scores = norm_qualities + 0.5 * np.array(uncertainties) * ucb_weights

    # Adaptive novelty penalty: scale based on stagnation and progress
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    x_vals = [cand['x'] for cand in context["pool"]]
  
    if len(x_vals) > 1:
        K_sim = np.zeros((len(context["pool"]), len(context["pool"])))
        
        # Compute pairwise similarities (Gaussian kernel)
        for i, xi in enumerate(x_vals):
            dists_sq = [np.sum((xi - xj)**2) for j, _ in enumerate(x_vals)]
            # Set self-similarity to zero
            dists_sq[i] = np.inf  
            
            K_sim[:,i] = np.exp(-0.5 * np.array(dists_sq))

        avg_similarity_to_observed = []
        
        if len(context["X_obs"]) > 1:
           X_obsv = context['X_obs']
           
           # For each candidate, compute similarity to all observed points
           for i in range(len(x_vals)):
               sim_scores = [K_sim[i][j] 
                             for j in range(K_sim.shape[0]) if np.all(X_obsv[j]==x_vals[i])]
              
               avg_similarity_to_observed.append(np.mean(sim_scores) if len(sim_scores)>0 else 1.)
        elif not context["X_obs"].size:
            # No observations yet
            avg_similarity_to_observed = [np.random.rand() for _ in range(len(context["pool"]))]
        
    else: 
         avg_similarity_to_observed = np.zeros_like(norm_qualities)
    
     novelty_penalty = 1. - (0.2 * np.array(avg_similarity_to_observed))

    # Combine all signals
    scores_final = ucb_scores * novelty_penalty

    return list(scores_final)