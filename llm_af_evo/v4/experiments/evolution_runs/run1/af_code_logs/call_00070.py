def score_pool(context):
    """Estimate HV expansion potential by sampling noisy GP predictions and compute expected gain from adding candidates to pareto front."""
    
    if not context["pool"]:
        return []
        
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    scores = []

    # Use a fixed noise level similar to parent A
    noise_level = 0.01
    
    num_samples_per_candidate = 64

    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        hv_gains = []
        
        # Sample from the joint posterior distribution of objectives 
        samples = np.zeros((num_samples_per_candidate, len(names)))
    
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]
            
            # Generate noisy sample
            noise_sample = np.random.normal(mean_val, std_val + noise_level)
            samples[:,i] = noise_sample
            
        
        hv_contributions = []
    
        for i in range(num_samples_per_candidate):
            new_point = samples[i]
          
            extended_front = np.vstack((context["pareto_front"],new_point))
            
            try:
                # Compute hypervolume of the expanded front
                hv_extended = hypervolume(extended_front, ref_point)
                
                # Subtract original HV to get gain from this sample point  
                hv_gain = max(hv_extended - hypervolume(context["pareto_front"],ref_point), 0.0)

            except:
                 hv_gain=0.

            hv_contributions.append(max(hv_gain / num_samples_per_candidate,1e-8))

        # Estimate expected HV gain
        exp_hv_gain = np.mean(np.array(hv_contributions))
        
        score = max(0.7 * cand["acq_value_norm"] + 0.3 * (exp_hv_gain), 
                    np.finfo(np.float32).eps)
                    
        scores.append(score)

    return [float(s) for s in scores]

def hypervolume(front, ref_point):
    
    if len(front.shape)==1:
       front =np.expand_dims(front,axis=0)
        
    n_points=len(front)
    vol_total=float(0.0)


    # For 2D case
    if n_points == 0 or (len(ref_point) != 2 and not all(x==y for x,y in zip(np.shape(front),[1,2]))):
        return float(vol_total)

    
    front = np.array(sorted([list(p) for p in front], key=lambda a: -a[0]))
    
    
   
    # For non-dominated points with two objectives
    if n_points == 1:
       vol_total= max( (ref_point[0] -front[0][0]) * 
                       ( ref_point[1]- front[0][1]), float(0.))
    
    else:  
        for i in range(n_points):
            x_i,y_i = front[i]
            
            if  y_i <= ref_point[1]:
                vol_total += max(ref_point[0] -x_i, 0.) * (ref_point[1]-y_i)
                
              
    

        
   
    return float(vol_total)