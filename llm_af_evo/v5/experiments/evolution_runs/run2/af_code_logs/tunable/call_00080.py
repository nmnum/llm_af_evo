def score_pool(context):
    """Estimates Pareto-optimality probability for each candidate via MC sampling, then scores based on density of high-probability candidates in feature space."""
    import numpy as np
    
    names = context["objective_names"]
    front = context["pareto_front"]
    
    n_samples = 25
    threshold_prob = 0.1
    sigma_novelty = 0.3

    # Precompute Pareto probabilities for all candidates once.
    cand_probs = []
    for i, cand in enumerate(context['pool']):
        gp_posterior = cand["gp_posterior"]
        
        samples = np.zeros((n_samples, len(names)))
        for j, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]
            # Assuming normal distribution and sample from it
            samples[:,j] = np.random.normal(mean_val, std_val, n_samples)
        
        count_pareto = 0.0
        
        for s in samples:
            is_dominated_by_front_point = False
            
            if len(front) == 0: 
                # If no front points yet - all are Pareto-optimal
                count_pareto += 1.
                continue
                
            for q in front:
                
                dominates_s = True  
                strictly_better_anywhere =False  

                for k, val_q in enumerate(q):
                    if not (val_q >= s[k]):
                        dominates_s=False 
                        break
                    
                    
                # If all are greater or equal and at least one is stricly better
                if dominates_s:
                     strict_comparison = any(val_q > s[k]  for k,val_q in enumerate(q))
                     
                     if strict_comparison:  
                         is_dominated_by_front_point=True   
                         break 
                         
            if not is_dominated_by_front_point :
                 count_pareto +=1.0
                
        prob = float(count_pareto) / n_samples
        cand_probs.append(prob)

    # Compute density scores based on high-probability candidates.
    
    x_pool= np.array([cand['x'] for cand in context["pool"]])
  
    score_list=[]
    
    
    for i, _  in enumerate(context["pool"]) :
        
      
       prob = float(cand_probs[i]) 
       
      # Use inverse distance weighted density
        if(prob<threshold_prob):
            final_score=0.1*prob   ## low weight to very unlikely points.
            
        else:
           dists=np.sum((x_pool-x_pool[i])**2,axis=-1)
          
          # Avoid division by zero and use small eps 
           
           inv_dists =  np.where(dists==0, float('inf'), (dists+1e-8)**(-3))
           

            
             ## Sum weights for candidates with prob > threshold
         
  
   
         density_score=np.sum(inv_dists * [cand_probs[j]>=threshold_prob 
                                               if i!=j else 0.0   # Skip self comparison.
                                              for j in range(len(cand_probs))]) / (np.sum([1.*((i==j) or cand_probs[j] >= threshold_prob)
                                                                                          for j in range(len(cand_probs))] )+1e-8)

        
            final_score =  prob + sigma_novelty * density_score

        score_list.append(final_score)


    return [float(score)for score in score_list ]