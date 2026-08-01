def score_pool(context):
    """
    Estimate hypervolume improvement potential by sampling candidates' objectives 
    from their GP posteriors and computing how much each would expand the current Pareto front.
    This approach directly optimizes for HV gain, unlike simple mean-based scores,
    while incorporating uncertainty via Monte Carlo estimation. The method favors
    candidates that are likely to be non-dominated (i.e., lie outside or near 
    existing fronts) under sampled objective values and penalizes those with low expected
    contribution based on current front shape.
    """
    import numpy as np
    
    # Number of samples for monte carlo estimation
    n_samples = 100

    names = context["objective_names"]
    
    scores = []
    pareto_front = context['pareto_front']
    ref_point = context['ref_point']

    def hypervolume_improvement(candidate_posterior, front_points):
        """Estimate HV improvement by sampling from candidate's GP posterior."""
        
        # Sample objectives for this candidate
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names): 
            mean_val = candidate_posterior[name]["mean"]
            std_val = candidate_posterior[name]["std"]  
            
            if not (np.isfinite(mean_val) and np.isfinite(std_val)):
                # Fallback to deterministic value
                samples[:,i] = mean_val
            else:
                samps = np.random.normal(loc=mean_val, scale=std_val, size=n_samples)
                # Clip extreme outliers for stability  
                lower_bound, upper_bound = (np.percentile(samps, 1), 
                                            np.percentile(samps, 99))
                
                samples[:,i] = np.clip(samps, a_min=lower_bound, a_max=upper_bound)

        hv_improvements = []
        
        for sample in samples:
            # For each sampled point from candidate's posterior
            new_front_points = [p.copy() for p in front_points]
            
            if not any(np.allclose(p,sample) or np.any(sample > ref_point)
                       for p in new_front_points):
                # If this is a non-dominated sample (relative to current front), 
                # add it and compute HV improvement
                
                valid_new = []
                
                dominates_any_existing = False
                dominated_by_others = []  # Points that are now dominated by `sample`
            
                for p in new_front_points:
                    if np.all(p >= sample) and not all(p == sample): 
                        # `p` is dominated or equal to the sampled point (i.e., dominates)
                        
                        valid_new.append(sample.copy())
                        dominate = True
                      
                      else:  # Not dominating this front member
                        
                          same_point = bool(np.allclose(s, p))  
                          
                            if not same_point:
                                invalid_points.add(p) 
                                
                    elif np.any(sample > p):
                         dominated_by_others.append(i)
                         
                new_front_points.extend(valid_new)

                
            else:  # Already in or worse than front (dominated), so skip
                 continue
                
            
             hv = compute_hypervolume(new_front_points, ref_point) 
             

        return np.mean(hv_improvements)


    for cand in context["pool"]:
        
       gp_posterior = cand['gp_posterior']
       
      # Compute the score using MC sampling to estimate HV gain
       try:
            hv_score = hypervolume_improvement(gp_posterior, pareto_front)
        except Exception as e: 
             print(f"Warning - error in computing candidate's HV improvement (likely numerical): {e}")
            hv_score = 0.0

        scores.append(hv_score)

    return scores


def compute_hypervolume(points, ref_point):
    
     # This is a placeholder for the actual hypervolume computation.
      if len(points) == 1:
          p = points[0]
           vol= np.prod([max(0.0,r - x)
                         for r,x in zip(ref_point,p)])
       else: 
            raise NotImplementedError("Multi-point HV calculation required here.")
            
    return float(vol)

# The actual implementation should use an efficient library like `pyhv` or similar
 # to avoid reimplementing hypervolume calculations from scratch.