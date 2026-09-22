def score_pool(context):
    """Estimates Pareto-optimality probability for each candidate via MC sampling, then scores based on local density among high-probability points."""
    np.random.seed(0)  # For reproducibility in sandbox
    
    n_samples = 25
    threshold_prob = 0.1
    names = context["objective_names"]
    
    pareto_front = context["pareto_front"] 
    front_range = context["pareto_front_range"]

    def dominates(q, s):
        return all(q[i] >= s[i] for i in range(len(s))) and any(q[i] > s[i] for i in range(len(s)))
        
    # Step 1: Compute Pareto probabilities
    pareto_probs = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        samples = np.zeros((n_samples, len(names))) 
        for idx,name in enumerate(names):
            mean,std=gp_posterior[name]["mean"],gp_posterior[name]["std"]  
            # Sample from normal distribution
            samples[:,idx] = np.random.normal(mean, std, n_samples)
            
        
        count_pareto = 0
        
        for s in samples:
            is_pareto_optimal=True
            
            for q in pareto_front: 
                if dominates(q,s):
                    is_pareto_optimal=False
                    break
                    
            if is_pareto_optimal:
                count_pareto +=1
                
                
        
        prob = float(count_pareto)/n_samples  
        pareto_probs.append(prob)
    
    # Step 2 : Compute density score based on high-probability candidates 
   
    scores=[]
    

    
    for i,cand in enumerate(context["pool"]):
        

        p=pareto_probs[i]
        
        if p<threshold_prob:
            final_score = -1e9
        else:  
            
      
            x_i=cand['x']
          
            distances=np.linalg.norm(np.array([c['x']for c in context['pool']])-x_i,axis=1)
       
 
   
           
 

  

    
     
 
            
                
                 
                    
                    if p_j > threshold_prob:
                        dist = max(0.001, d)  # Avoid division by zero
                      
                        
                         
                            
                                
                                    
                                        
                                            
                                                
                                  
                                        inv_dist += (p_j /dist )
                           
            final_score=inv_dist
            
        scores.append(final_score)
        
    return scores