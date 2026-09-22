def score_pool(context):
    """Estimates hypervolume improvement by bootstrapping Pareto fronts and measuring candidate impact on each."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    # Bootstrap resampling of Y_obs to estimate front uncertainty
    n_bootstrap = 15
    bootstrap_improvements = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement from observed outcomes (Y_obs)
        indices = np.random.choice(len(context['Y_obs']), size=len(context['Y_obs']), replace=True)
        Y_sampled = context["Y_obs"][indices]
        
        # Compute non-dominated set using vectorized dominance check
        n_points = len(Y_sampled)
        dominated = np.zeros(n_points, dtype=bool)

        for i in range(n_points):
            for j in range(i + 1, n_points):
                if not dominated[i] and all(Y_sampled[j][k] >= Y_sampled[i][k] for k in range(len(names))) and any(Y_sampled[j][k] > Y_sampled[i][k] for k in range(len(names))):
                    dominated[i] = True
                elif not dominated[j] and all(Y_sampled[i][k] >= Y_sampled[j][k] for k in range(len(names))) and any(Y_sampled[i][k] > Y_sampled[j][k] for k in range(len(names))):
                    dominated[j] = True
        
        # Get non-dominated points
        pf_indices = np.where(~dominated)[0]
        if len(pf_indices) == 0:
            continue
            
        sampled_pf = Y_sampled[pf_indices]

        candidate_improvements = []
        
        for cand in context["pool"]:
            gp_pred = cand["gp_posterior"]
            
            # Predicted objectives
            pred_obj = [gp_pred[name]["mean"] for name in names]
                        
            if len(sampled_pf) == 0:
                hv_before_adding = 0.0
            else:
                try: 
                    hypervolume_before = compute_hypervolume(sampled_pf, ref_point)
                except Exception as e:
                    # Fallback to zero or minimal volume estimate on error  
                    hypervolume_before = 0.

            
            if len(sampled_pf) == 0 and all(pred_obj[k] >= ref_point[k] for k in range(len(names))):
                 hv_after_adding = np.prod([pred_obj[k]-ref_point[k] for k in range(len(names))])
            elif len(sampled_pf) > 0:
                pf_plus_candidate = np.vstack((sampled_pf, pred_obj))
                
                try: 
                    hypervolume_after= compute_hypervolume(pf_plus_candidate, ref_point)
                except Exception as e:
                     # On error fallback to minimal volume
                      hypervolume_after = hypervolume_before
                    
            else:
                 hv_after_adding = 0.
            
        
            improvement = max(0., (hypervolume_after - hypervolume_before))
                
            candidate_improvements.append(improvement)
    
        bootstrap_improvements.extend(candidate_improvements)

    # Return average across all resamples
    scores= [np.mean([bootstrap_improvements[i] for i in range(j, len(bootstrap_improvements),len(context["pool"]))]) 
             if j < len(context['pool']) else 0.  
            for j in range(len(context["pool"]))]
    
    return scores

def compute_hypervolume(points, ref_point):
     import numpy as np
     # Simplified hypervolume computation using reference point and points (assumes all maximized)
     if len(points) == 0:
        raise ValueError("No valid set of points to calculate HV.")
        
     d = len(ref_point)

    try: 
         hv_sum=1.
         
          for i in range(len(points)):
            prod_term = np.prod([max(0., ref_point[k] - points[i][k])  
                                 if k < min(d, 2) else (ref_point[0]-points[i][0])
                               for k in range(min(d,len(ref_point))) 
                            ])
            
            hv_sum += max(prod_term ,1e-3)
        
         return float(hv_sum/len(points)) # Normalize by number of points
    except:
        raise ValueError("Invalid hypervolume calculation.")