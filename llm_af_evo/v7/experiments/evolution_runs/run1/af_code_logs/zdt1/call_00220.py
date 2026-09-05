def modifier(context):
    """Boosts candidates that are likely to dominate existing front points, based on posterior sampling and dynamic thresholding."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Early exit if no observations yet 
    progress = context["campaign"]["progress"]  
    if len(context.get("X_obs", [])) == 0 or progress < 0.1:
        return [0.] * len(context["pool"])
        
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Sample from posteriors to estimate potential dominance
    n_samples = 256 
    domination_scores = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        means = [gp_posterior[name]["mean"] for name in names]
        stds = [gp_posterior[name]["std"] for name in names]

        # Generate samples from the joint posterior
        cov_matrix = np.diag(np.array(stds)**2)
        try:
            sampled_means = np.random.multivariate_normal(means, cov_matrix, n_samples) 
        except:  # fallback if covariance is ill-conditioned  
            sampled_means = np.tile(means, (n_samples,1))
        
        domination_count = 0
        for sample in sampled_means:
            
            candidate_point = np.array(sample)
                
            front_points = context["pareto_front"]
                        
            # Check how many current Pareto points this point dominates 
            dominated_by_pf = False
            
            for pf_point in front_points: 

                if all(pf_point <= candidate_point):  # Dominated by PF
                    dominated_by_pf = True  
                    break
                    
                    
            
            if not dominated_by_pf:
                 domination_count +=1

        dominance_ratio = float(domination_count) / n_samples
        
        # Scale score based on how much it's likely to dominate 
        scaled_score = np.log2( 1 + (dominance_ratio * 5.0)) 

        
        domination_scores.append(scaled_score)

    max_dom Score =np.max(domination_scores)
    
    if max_domScore <= 0:
         return [0.] * len(context["pool"])
         
   
    # Normalize and apply a decay factor based on progress
    normalized_scores= []
    for score in domination_scores: 
        norm_val = (score /max_dom Score)  
        
        # Decay as campaign progresses to favor exploitation later
        weight_decay = 1.0 - context["campaign"]["progress"] 
        
        final_score =norm_val *weight_decay
        
        normalized_scores.append(final_score)

    return [val for val in normalized_scores]