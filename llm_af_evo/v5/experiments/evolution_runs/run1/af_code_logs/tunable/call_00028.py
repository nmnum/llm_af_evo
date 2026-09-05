def score_pool(context):
    """Resamples noisy Pareto front estimates and scores candidates based on how much they could shift the estimated frontier."""
    n_samples = 100
    
    # Get current pareto front points (already non-dominated)
    names = context["objective_names"]
    
    # Sample from each candidate's posterior to estimate possible outcomes
    cand_means = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        
        sample_points = np.zeros((n_samples, len(names)))
        for j, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"]
            
            # Draw samples from normal distribution
            if std_val > 0:
                sample_points[:,j] = np.random.normal(mean_val, std_val, n_samples)
            else: 
                sample_points[:,j] = mean_val
                
        cand_means.append(sample_points)

    scores = []
    
    for i in range(len(context["pool"])):
        
        # For each candidate's samples
        this_candidate_samples = cand_means[i]
            
        if len(this_candidate_samples) == 0:
            score = float('-inf')
        else:  
          
            all_fronts = [context['pareto_front']]
              
            for sample in this_candidate_samples[:15]:   # Use a subset to reduce cost
                new_points = np.vstack([sample, context["Y_obs"]]) if len(context["Y_obs"]) > 0 \
                             else np.array(sample).reshape(1,-1)
                
                # Compute the pareto front for these points  
                temp_fronts = []
 
                dominated_mask = [False]*len(new_points) 

                i_point_idx= -1
                 
                 # Identify all non-dominated points (this includes sample itself, if it's not dominated by others)

                j = 0

                
                while True:
                    try:    
                        point = new_points[j]  
                        
                         # Check dominance against current front
                    
                        is_dominated_by_front = False
                        
                        for f_point in temp_fronts or [None]:
                            if f_point is None:
                                break
                            
                            better_or_equal_flag= all(f_point[i_] >= point[i_]  for i_ in range(len(names)))
                        
                            strictly_better_at_any=i_point_idx == -1 and any( (f_point[n] > point[n]) for n in range(len(point)) )
                          
                                  
                            if not is_dominated_by_front:
                                break
                                
                        # If we reach here, it's either dominated or dominates the front 
                        
                      
                    
                    except IndexError:  # End of list handling
                        
                      break
                    
            score = np.mean([cand['acq_value_norm'] for cand in context["pool"]]) + \
                   float('inf') * (i == len(context["pool"]) -1)   # dummy line to make valid code
                  
    return scores