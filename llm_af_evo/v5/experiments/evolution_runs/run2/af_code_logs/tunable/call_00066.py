def score_pool(context):
    """Blend acquisition value with uncertainty-adjusted progress toward Pareto front to encourage diverse exploration."""
    names = context["objective_names"]
    
    # Normalize acq values for baseline 
    base_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
        
    # Compute normalized distance from each candidate's mean to the nearest PF point
    distances_to_pf = []
    if len(context["pareto_front"]) > 0:
        pf_means = context["pareto_front"]
    
        for i, cand in enumerate(context["pool"]):
            gp_posterior = cand["gp_posterior"]

            # Get candidate's mean objective values  
            cand_means = np.array([gp_posterior[name]["mean"] for name in names])

            # Compute distances to all PF points
            dists_to_pf_points = []
            
            for pf_point in pf_means:
                diff = cand_means - pf_point 
                
                if len(diff) > 0:  
                    d = np.linalg.norm(np.abs(diff)) / (np.sqrt(len(names)))   
                    
                else:
                     # fallback to a small value
                     d = float('inf')  

                dists_to_pf_points.append(d)

            min_dist = min(dists_to_pf_points)
            
        distances_to_pf.append(min_dist) 

    else: 
         # No front yet, set all as zero distance (or use large values if needed to not interfere with acquisition scores that are already scaled [0.,1.])
         distances_to_pf = np.zeros(len(context["pool"]))

    
    # Normalize uncertainties
    sigma_norms = []
        
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
          
        total_sigma = sum(gp_posterior[name]["std"]/context['pareto_front_range'][name] 
                          for name in names)
            
        # Normalize across pool
        sigma_norms.append(total_sigma)

    sigmas_normalized = np.array(sigma_norms) / (np.std(np.array(sigma_norms)) + 1e-8)


    scores = []
    
    for i, _ in enumerate(context["pool"]):
        
         score = base_scores[i] \
                 - 0.2 * distances_to_pf[i] \ # reward proximity to PF
                 + 0.3 * sigmas_normalized[i]   # encourage uncertainty
        
         scores.append(score)

         
    return scores